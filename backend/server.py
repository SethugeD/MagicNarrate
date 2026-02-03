import os
import io
import re
import torch
import base64
import soundfile as sf
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from torchvision import transforms
import google.generativeai as genai
from transformers import AutoTokenizer, set_seed
from parler_tts import ParlerTTSForConditionalGeneration

# Load environment variables
load_dotenv()

# Import your model (matches your Colab!)
from model_def import CaptionModel, get_resnet_extractor

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
# Use MPS (Apple Silicon GPU) if available, otherwise CUDA, otherwise CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🚀 Using MPS (Apple Silicon GPU) for captioning")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("🚀 Using CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ Using CPU (slower)")

# TTS must run on CPU - MPS doesn't support large channel counts in audio decoder
TTS_DEVICE = torch.device("cpu")
print("🔊 TTS will run on CPU (MPS limitation)")

# Available speakers for TTS (parler-tts-mini-expresso voices)
AVAILABLE_SPEAKERS = ["Jon", "Lea", "Gary", "Jenna"]

# --- LOAD MODELS ---
print("Loading Models...")

# 1. Load Vocabulary (your Colab saves as (word2idx, idx2word) tuple)
try:
    vocab_path = os.path.join("image_captioning", "vocab.pt")
    word2idx, idx2word = torch.load(vocab_path, map_location="cpu")
    vocab_size = len(word2idx)
    print(f"Vocabulary loaded. Size: {vocab_size}")
except Exception as e:
    print(f"Error loading vocab: {e}")
    word2idx, idx2word = {}, {}
    vocab_size = 0

# 2. Load ResNet for feature extraction 
print("Loading ResNet50 feature extractor...")
resnet = get_resnet_extractor().to(DEVICE)

# 3. Load your trained CaptionModel
print("Loading Caption Model...")
model = CaptionModel(vocab_size).to(DEVICE)

model_path = os.path.join("image_captioning", "resnet50_model.pth")
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    print("Caption model loaded successfully!")
else:
    print(f"WARNING: Model file not found at {model_path}")

model.eval()

# 4. Load Parler-TTS (using expresso model for better emotional expression)
print("Loading TTS...")
tts_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-expresso").to(TTS_DEVICE)
tts_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-expresso")

# Image Transforms 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
])

print("All models loaded! Server ready.")


def split_story(text: str, max_sentences: int = 2) -> list:
    """Split story into chunks of N sentences to avoid TTS failure"""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = " ".join(sentences[i:i + max_sentences])
        if chunk:
            chunks.append(chunk)
    return chunks


# Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
llm = genai.GenerativeModel("gemma-3-4b-it")


def generate_story_text(emotion: str, genre: str, sentence: str) -> str:
    """Generate story using Gemini API with the improved prompt"""
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
    response = llm.generate_content(prompt)
    return response.text


# Map UI tones to parler-tts-mini-expresso emotions
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
    # Convert UI tone to parler-tts emotion
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
        print(f"🔊 Generating audio for part {idx+1}/{len(story_chunks)}")
        
        prompt_tokens = tts_tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True
        )
        prompt_input_ids = prompt_tokens.input_ids.to(TTS_DEVICE)
        prompt_attention_mask = prompt_tokens.attention_mask.to(TTS_DEVICE)
        
        with torch.no_grad():
            audio = tts_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask
            )
        
        audio_np = audio.cpu().numpy().squeeze()
        all_audio.append(audio_np)
        
        # Pause between chunks (0.4 sec)
        pause = np.zeros(int(0.4 * tts_model.config.sampling_rate))
        all_audio.append(pause)
    
    # Concatenate all audio
    final_audio = np.concatenate(all_audio)
    
    # Convert to base64
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
    """Generate a story from a text prompt (no image needed)"""
    print(f"Generating {genre} story from text prompt with speaker: {speaker}...")
    
    # 1. STORY GENERATION (Gemini)
    story_text = generate_story_text(tone, genre, prompt)
    print(f"Generated story: {story_text[:100]}...")

    # 2. AUDIO GENERATION
    audio_b64, duration = generate_audio(story_text, tone, speaker)

    return {
        "caption": prompt,
        "story": story_text,
        "audio": f"data:audio/wav;base64,{audio_b64}",
        "duration": duration
    }

@app.post("/generate")
async def generate_story(
    image: UploadFile = File(...),
    genre: str = Form("fantasy"),
    tone: str = Form("joyful"),
    speaker: str = Form("Lea")
):
    print(f"Processing image for {genre} story with speaker: {speaker}...")
    
    # 1. CAPTIONING
    image_bytes = await image.read()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        # Extract features using ResNet (same as your Colab)
        features = resnet(img_tensor).squeeze()  # [2048]
        
        # Generate caption with beam search
        caption = model.generate_caption(features, idx2word, max_len=20, device=DEVICE, word2idx=word2idx)
    
    print(f"Caption: {caption}")

    # 2. STORY GENERATION (Gemini)
    story_text = generate_story_text(tone, genre, caption)
    print(f"Generated story: {story_text[:100]}...")

    # 3. AUDIO GENERATION
    audio_b64, duration = generate_audio(story_text, tone, speaker)

    return {
        "caption": caption,
        "story": story_text,
        "audio": f"data:audio/wav;base64,{audio_b64}",
        "duration": duration
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)