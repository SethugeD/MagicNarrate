import os
import io
import re
import torch
import base64
import warnings
import soundfile as sf
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
from openai import OpenAI
from transformers import AutoTokenizer, set_seed
from parler_tts import ParlerTTSForConditionalGeneration

load_dotenv()

from model_def import DecoderWithAttention, get_resnet_extractor, extract_features

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

TTS_DEVICE = torch.device("cpu")

AVAILABLE_SPEAKERS = ["Jon", "Lea", "Gary", "Jenna"]

print("Loading models...")

try:
    vocab_path = os.path.join("image_captioning", "vocab.pt")
    word2idx, idx2word = torch.load(vocab_path, map_location="cpu")
    vocab_size = len(word2idx)
    print(f"Vocabulary loaded: {vocab_size} words")
except Exception as e:
    print(f"Error loading vocabulary: {e}")
    word2idx, idx2word = {}, {}
    vocab_size = 0

resnet = get_resnet_extractor().to(DEVICE)
model = DecoderWithAttention(vocab_size).to(DEVICE)

model_path = os.path.join("image_captioning", "resnet50_attention_model.pth")
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    print("Caption model loaded")
else:
    print(f"Warning: Model file not found at {model_path}")

model.eval()

tts_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-expresso").to(TTS_DEVICE)
tts_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-expresso")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Models loaded. Server ready.")


def split_story(text: str, max_sentences: int = 2) -> list:
    """Split story into chunks of N sentences to avoid TTS failure"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        if chunk:
            chunks.append(chunk)
    return chunks


client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def generate_story_text(emotion: str, genre: str, sentence: str) -> str:
    """Generate story using OpenAI API with the improved prompt"""
    prompt = f"""
You are a creative children's storyteller.

Emotion: {emotion}
Genre: {genre}

Task:
Create a short story (80–100 words) suitable for children.
The story must strongly reflect the given emotion through:
- word choice
- sentence rhythm
- atmosphere
- character reactions

IMPORTANT RULES:
- Do NOT use sound effects or onomatopoeia
- Avoid exaggerated punctuation (!!!, ???)
- Use calm, natural narrative sentences suitable for text-to-speech systems

Starting idea:
"{sentence}"

End the story on an emotionally meaningful note.
"""
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


TONE_TO_EMOTION = {
    "joyful": "happy",
    "funny": "laughing",
    "mysterious": "whisper",
    "confused": "confused",
    "calm": "default",
    "sad": "sad",
    "dramatic": "emphasis"
}

def generate_audio(story_text: str, emotion: str, speaker: str = "Lea"):
    """Generate TTS audio with chunking for better quality"""
    tts_emotion = TONE_TO_EMOTION.get(emotion, "default")
    
    description = f"""
    {speaker} speaks slowly in a {tts_emotion} tone
    with emphasis and high quality audio.
    """
    
    desc_tokens = tts_tokenizer(
        description,
        return_tensors="pt",
        truncation=True
    )
    input_ids = desc_tokens.input_ids.to(TTS_DEVICE)
    attention_mask = desc_tokens.attention_mask.to(TTS_DEVICE)
    
    story_chunks = split_story(story_text, max_sentences=2)
    all_audio = []
    
    set_seed(42)
    
    for idx, chunk in enumerate(story_chunks):
        print(f"Generating audio for part {idx+1}/{len(story_chunks)}")
        
        prompt_tokens = tts_tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True
        )
        prompt_input_ids = prompt_tokens.input_ids.to(TTS_DEVICE)
        prompt_attention_mask = prompt_tokens.attention_mask.to(TTS_DEVICE)
        
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                audio = tts_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask
                )
        
        audio_np = audio.cpu().numpy().squeeze()
        all_audio.append(audio_np)
        
        pause = np.zeros(int(0.4 * tts_model.config.sampling_rate))
        all_audio.append(pause)
    
    final_audio = np.concatenate(all_audio)
    buffer = io.BytesIO()
    sf.write(buffer, final_audio, tts_model.config.sampling_rate, format='WAV')
    audio_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    duration = len(final_audio) / tts_model.config.sampling_rate
    return audio_b64, duration


@app.get("/speakers")
async def get_speakers():
    """Return available speaker voices"""
    return {"speakers": AVAILABLE_SPEAKERS}

@app.post("/generate-from-text")
async def generate_story_from_text(
    prompt: str = Form(...),
    genre: str = Form("fantasy"),
    tone: str = Form("joyful"),
    speaker: str = Form("Lea")
):
    """Generate a story from a text prompt"""
    print(f"Generating {genre} story with speaker: {speaker}")
    
    story_text = generate_story_text(tone, genre, prompt)
    print(f"Story generated: {story_text[:100]}...")
    return {
        "caption": prompt,
        "story": story_text,
    }

@app.post("/generate")
async def generate_story(
    image: UploadFile = File(...),
    genre: str = Form("fantasy"),
    tone: str = Form("joyful"),
    speaker: str = Form("Lea")
):
    print(f"Processing image for {genre} story with speaker: {speaker}")
    
    image_bytes = await image.read()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        features = extract_features(resnet, img_tensor, device=DEVICE)
        caption = model.generate_caption(features, idx2word, max_len=50, device=DEVICE, word2idx=word2idx, beam_size=5)
    
    print(f"Caption: {caption}")
    
    story_text = generate_story_text(tone, genre, caption)
    print(f"Story generated: {story_text[:100]}...")
    return {
        "caption": caption,
        "story": story_text,
    }


@app.post("/generate-audio")
async def generate_audio_endpoint(
    story: str = Form(...),
    tone: str = Form("joyful"),
    speaker: str = Form("Lea")
):
    """Generate audio narration for a given story text"""
    print(f"Generating audio with speaker: {speaker}, tone: {tone}")
    
    audio_b64, duration = generate_audio(story, tone, speaker)
    
    return {
        "audio": f"data:audio/wav;base64,{audio_b64}",
        "duration": duration
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)