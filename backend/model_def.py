import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import random

# Special token indices (must match vocab)
PAD_IDX = 0
START_IDX = 1
END_IDX = 2


# -------------------------
# ATTENTION MECHANISM
# -------------------------
class Attention(nn.Module):
    """Additive (Bahdanau) Attention"""
    def __init__(self, encoder_dim, decoder_dim, attn_dim):
        super().__init__()
        self.W_enc = nn.Linear(encoder_dim, attn_dim)
        self.W_dec = nn.Linear(decoder_dim, attn_dim)
        self.V = nn.Linear(attn_dim, 1)

    def forward(self, encoder_out, h):
        # encoder_out: (B, 49, encoder_dim)  h: (B, decoder_dim)
        e = self.W_enc(encoder_out)              # (B, 49, attn_dim)
        d = self.W_dec(h).unsqueeze(1)           # (B, 1,  attn_dim)
        score = self.V(torch.tanh(e + d))        # (B, 49, 1)
        alpha = torch.softmax(score, dim=1)      # (B, 49, 1)
        context = (alpha * encoder_out).sum(1)   # (B, encoder_dim)
        return context, alpha.squeeze(-1)        # (B, encoder_dim), (B, 49)


# -------------------------
# DECODER WITH ATTENTION
# -------------------------
class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, encoder_dim=2048,
                 decoder_dim=384, attn_dim=256, dropout=0.6):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.attention = Attention(encoder_dim, decoder_dim, attn_dim)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def _init_hidden(self, encoder_out):
        """Initialize hidden state from mean of encoder outputs"""
        mean_enc = encoder_out.mean(dim=1)       # (B, encoder_dim)
        h = torch.tanh(self.init_h(mean_enc))    # (B, decoder_dim)
        c = torch.tanh(self.init_c(mean_enc))
        return h, c

    def forward(self, encoder_out, captions, teacher_forcing_ratio=1.0):
        """
        encoder_out: (B, 49, encoder_dim)
        captions: (B, T) - full sequence including <start> and <end>
        returns: logits (B, T-1, vocab_size)
        """
        B, T = captions.size()
        h, c = self._init_hidden(encoder_out)

        outputs = []
        input_token = captions[:, 0]             # always <start>

        for t in range(T - 1):                   # predict tokens 1 … T-1
            emb = self.dropout(self.embedding(input_token))  # (B, embed_dim)
            context, _ = self.attention(encoder_out, h)      # (B, enc_dim)
            lstm_in = self.dropout(torch.cat([emb, context], dim=1))  # (B, E+enc)
            h, c = self.lstm(lstm_in, (h, c))
            logits = self.fc(self.dropout(h))                # (B, V)
            outputs.append(logits)

            # teacher forcing: use ground-truth or model prediction
            use_gt = random.random() < teacher_forcing_ratio
            input_token = captions[:, t + 1] if use_gt else logits.argmax(1)

        return torch.stack(outputs, dim=1)       # (B, T-1, V)

    def generate_caption(self, features, idx2word, max_len=50, device="cpu", 
                         word2idx=None, beam_size=5):
        """
        Generate caption using beam search.
        features: (49, 2048) tensor for a single image
        """
        self.eval()
        
        with torch.no_grad():
            # Build word2idx if not provided
            if word2idx is None:
                word2idx = {word: idx for idx, word in idx2word.items()}
            
            # Determine start/end indices
            start_idx = word2idx.get('<start>', START_IDX)
            end_idx = word2idx.get('<end>', END_IDX)
            pad_idx = word2idx.get('<pad>', PAD_IDX)
            
            # Ensure features are (1, 49, 2048)
            if features.dim() == 2:
                enc = features.unsqueeze(0).to(device)  # (1, 49, 2048)
            else:
                enc = features.to(device)
            
            h, c = self._init_hidden(enc)  # (1, decoder_dim)

            # each beam: [sequence of token ids, log-prob, h, c]
            beams = [([start_idx], 0.0, h.clone(), c.clone())]
            completed = []

            for _ in range(max_len):
                new_beams = []
                for seq, score, h_beam, c_beam in beams:
                    tok = torch.tensor([seq[-1]], device=device)
                    emb = self.embedding(tok)                     # (1, embed_dim)
                    ctx, _ = self.attention(enc, h_beam)
                    lstm_in = torch.cat([emb, ctx], dim=1)
                    h_new, c_new = self.lstm(lstm_in, (h_beam, c_beam))
                    logits = self.fc(self.dropout(h_new))        # (1, V)

                    # mask <pad> so it can never be chosen
                    logits[0, pad_idx] = -1e9

                    log_probs = F.log_softmax(logits, dim=-1)  # (1, V)
                    topk_lp, topk_idx = log_probs[0].topk(beam_size)

                    for lp, idx in zip(topk_lp.tolist(), topk_idx.tolist()):
                        new_seq = seq + [idx]
                        new_score = score + lp
                        if idx == end_idx:
                            completed.append((new_seq, new_score))
                        else:
                            new_beams.append((new_seq, new_score, h_new.clone(), c_new.clone()))

                # keep top beams
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
                if not beams:
                    break

            # add remaining beams as partial completions
            completed.extend([(seq, score) for seq, score, _, _ in beams])

            if not completed:
                return "Caption generation failed."

            # pick highest-scoring caption (length-normalized)
            best_seq = max(completed, key=lambda x: x[1] / len(x[0]))[0]

            # convert to words, skip <start>, <end>, <pad>
            caption_words = []
            for idx in best_seq:
                word = idx2word.get(idx, '')
                if word in {'<start>', '<end>', '<pad>', ''}:
                    continue
                caption_words.append(word)

            return " ".join(caption_words)


# -------------------------
# RESNET FEATURE EXTRACTOR (for Attention model)
# -------------------------
def get_resnet_extractor():
    """
    Returns ResNet50 encoder that outputs spatial features (B, 2048, 7, 7).
    For attention model: remove last 2 layers (avgpool and fc).
    """
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Remove avgpool and fc layers to keep spatial information
    encoder = nn.Sequential(*list(resnet.children())[:-2])
    encoder.eval()
    
    # Freeze parameters
    for p in encoder.parameters():
        p.requires_grad = False
    
    return encoder


def extract_features(encoder, img_tensor, device="cpu"):
    """
    Extract features from image tensor for attention model.
    img_tensor: (B, 3, 224, 224) or (3, 224, 224)
    Returns: (B, 49, 2048) or (49, 2048) tensor
    """
    single_image = img_tensor.dim() == 3
    if single_image:
        img_tensor = img_tensor.unsqueeze(0)
    
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        feat = encoder(img_tensor)  # (B, 2048, 7, 7)
        # Reshape to (B, 49, 2048) for attention
        feat = feat.permute(0, 2, 3, 1).reshape(feat.size(0), 49, 2048)
    
    if single_image:
        return feat.squeeze(0)  # (49, 2048)
    return feat  # (B, 49, 2048)