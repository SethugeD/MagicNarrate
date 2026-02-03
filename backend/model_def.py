import torch
import torch.nn as nn
import torchvision.models as models

# -------------------------
# CAPTION MODEL DEFINITION
# -------------------------
class CaptionModel(nn.Module):
    def __init__(self, vocab_size, feature_size=2048, embed_size=256, hidden_size=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.feature_fc = nn.Linear(feature_size, hidden_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        # features: [batch, 2048]
        h0 = self.feature_fc(features).unsqueeze(0)  # [1, batch, hidden]
        c0 = torch.zeros_like(h0)

        embeddings = self.embedding(captions)  # [batch, seq_len, embed]
        outputs, _ = self.lstm(embeddings, (h0, c0))
        outputs = self.fc(outputs)
        return outputs

    def generate_caption(self, features, idx2word, max_len=20, device="cpu", word2idx=None, beam_size=3):
        import torch.nn.functional as F
        
        self.eval()
        
        with torch.no_grad():
            # features should be [2048]
            if features.dim() == 2:
                features = features.squeeze(0)
            
            # Initialize hidden state from image features
            h = self.feature_fc(features).unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
            c = torch.zeros_like(h).to(device)
            
            # Build word2idx if not provided
            if word2idx is None:
                word2idx = {word: idx for idx, word in idx2word.items()}
            
            # Start tokens: <, start, >
            start_tokens = [word2idx.get('<', 1), word2idx.get('start', 2), word2idx.get('>', 3)]
            beams = [(start_tokens, 0.0, h, c)]  # (sequence_of_ids, log_prob_score, hidden_state, cell_state)
            
            final_captions = []
            
            for _ in range(max_len):
                if not beams:
                    break
                
                new_beams = []
                for seq, score, h_state, c_state in beams:
                    # Check if this beam already ended
                    if idx2word.get(seq[-1], '') == 'end':
                        final_captions.append((seq, score))
                        continue
                    
                    last_word_tensor = torch.tensor([[seq[-1]]]).to(device)
                    emb = self.embedding(last_word_tensor)
                    out, (h_next, c_next) = self.lstm(emb, (h_state, c_state))
                    logits = self.fc(out.squeeze(1))
                    probs = F.log_softmax(logits, dim=1)
                    
                    topk_probs, topk_idx = probs.topk(beam_size)
                    
                    for i in range(beam_size):
                        next_word_idx = topk_idx[0][i].item()
                        new_score = score + topk_probs[0][i].item()
                        new_seq = seq + [next_word_idx]
                        
                        if idx2word.get(next_word_idx, '') == 'end':
                            final_captions.append((new_seq, new_score))
                        else:
                            new_beams.append((new_seq, new_score, h_next, c_next))
                
                # Keep top beam_size beams
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            
            # Add remaining beams as completed captions
            final_captions.extend([(seq, score) for seq, score, _, _ in beams])
            
            if not final_captions:
                return "Caption generation failed."
            
            # Get the best caption
            best_seq = sorted(final_captions, key=lambda x: x[1], reverse=True)[0][0]
            
            # Filter out special tokens: <, start, >, end
            caption_words = []
            for idx in best_seq:
                word = idx2word.get(idx, '')
                if word in {'<', 'start', '>', 'end', '<pad>', ''}:
                    continue
                caption_words.append(word)
            
            return " ".join(caption_words)


# -------------------------
# RESNET FEATURE EXTRACTOR
# -------------------------
def get_resnet_extractor():
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    # Remove the final classification layer
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.eval()
    
    # Freeze parameters
    for p in resnet.parameters():
        p.requires_grad = False
    
    return resnet