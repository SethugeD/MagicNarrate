import os
import io
import re
import uuid
import time
import asyncio
import torch
import base64
import warnings
import httpx
import soundfile as sf
import numpy as np
from typing import Any
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
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

TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "local").strip().lower()
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "").strip()
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "").strip()
RUNPOD_BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}" if RUNPOD_ENDPOINT_ID else ""
RUNPOD_POLL_INTERVAL_SEC = float(os.environ.get("AUDIO_JOB_POLL_INTERVAL_SEC", "2.5"))
AUDIO_JOB_TIMEOUT_SEC = int(os.environ.get("AUDIO_JOB_TIMEOUT_SEC", "420"))

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

tts_model = None
tts_tokenizer = None
if TTS_PROVIDER == "local":
    tts_model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-expresso").to(TTS_DEVICE)
    tts_tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-expresso")
else:
    print("Skipping local TTS model load because TTS_PROVIDER is set to runpod")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Models loaded. Server ready.")

audio_jobs: dict[str, dict[str, Any]] = {}


@app.get("/")
async def root():
    """Basic root route so the API base URL returns a helpful response."""
    return {
        "status": "ok",
        "service": "MagicNarrate backend",
        "message": "API is running. Use /speakers, /generate-from-text, /generate, /generate-audio-job.",
    }


@app.get("/health")
async def health():
    """Lightweight health check endpoint for deployment verification."""
    return {"status": "healthy"}


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
    if tts_model is None or tts_tokenizer is None:
        raise RuntimeError("Local TTS model is not loaded. Set TTS_PROVIDER=local.")

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


def _runpod_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }


def _normalize_audio_value(audio_value: Any) -> str:
    if not isinstance(audio_value, str) or not audio_value:
        return ""
    if audio_value.startswith("data:audio"):
        return audio_value
    if audio_value.startswith("http://") or audio_value.startswith("https://"):
        return audio_value
    return f"data:audio/wav;base64,{audio_value}"


def _extract_runpod_output(payload: dict[str, Any]) -> tuple[str, float, str]:
    output = payload.get("output")
    if output is None and isinstance(payload.get("data"), dict):
        output = payload["data"].get("output")

    if isinstance(output, list) and output:
        output = output[0]

    if not isinstance(output, dict):
        if isinstance(output, str):
            return _normalize_audio_value(output), 0.0, ""
        return "", 0.0, payload.get("error", "RunPod job completed without an audio output.")

    audio_candidate = (
        output.get("audio")
        or output.get("audio_base64")
        or output.get("wav_base64")
        or output.get("audio_url")
    )
    audio = _normalize_audio_value(audio_candidate)
    duration = float(output.get("duration", 0.0) or 0.0)
    error = output.get("error", "")
    if not audio and not error:
        error = payload.get("error", "RunPod completed but no audio payload was returned.")
    return audio, duration, error


def _status_message(status: str) -> str:
    return {
        "queued": "Queued on GPU",
        "running": "Generating audio",
        "done": "Audio ready",
        "failed": "Audio generation failed",
    }.get(status, "Processing")


def _compose_progress_message(job: dict[str, Any], status: str, payload: dict[str, Any] | None = None) -> str:
    if status == "queued":
        queue_position = None
        if isinstance(payload, dict):
            queue_position = payload.get("queuePosition") or payload.get("queue_position")
        if queue_position is not None:
            return f"Queued on GPU (position {queue_position})"

        expected_chunks = int(job.get("expected_chunks", 0) or 0)
        if expected_chunks > 0:
            return f"Queued on GPU (story split into {expected_chunks} chunks)"
        return _status_message("queued")

    if status == "running":
        expected_chunks = int(job.get("expected_chunks", 0) or 0)
        if expected_chunks <= 0:
            return _status_message("running")

        running_polls = int(job.get("running_polls", 0) or 0)
        current_chunk = max(1, min(expected_chunks, running_polls))
        return f"Generating audio: chunk {current_chunk}/{expected_chunks} (estimated)"

    return _status_message(status)


def _map_runpod_status(raw_status: str) -> str:
    status = (raw_status or "").upper()
    if status in {"IN_QUEUE", "QUEUED"}:
        return "queued"
    if status in {"IN_PROGRESS", "RUNNING"}:
        return "running"
    if status in {"COMPLETED", "SUCCEEDED"}:
        return "done"
    if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
        return "failed"
    return "running"


async def _submit_runpod_job(story: str, tone: str, speaker: str) -> tuple[str, str, str]:
    if not RUNPOD_ENDPOINT_ID or not RUNPOD_API_KEY:
        raise RuntimeError("RunPod is not configured. Set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID.")

    payload = {
        "input": {
            "story": story,
            "tone": tone,
            "speaker": speaker,
        }
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(f"{RUNPOD_BASE_URL}/run", headers=_runpod_headers(), json=payload)
        response.raise_for_status()
        data = response.json()

    external_id = data.get("id")
    if not external_id:
        raise RuntimeError("RunPod did not return a job id.")

    status = _map_runpod_status(data.get("status", "IN_QUEUE"))
    return external_id, status, _status_message(status)


async def _refresh_runpod_job(job: dict[str, Any]) -> None:
    external_id = job.get("provider_job_id")
    if not external_id:
        return

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{RUNPOD_BASE_URL}/status/{external_id}", headers=_runpod_headers())
        response.raise_for_status()
        payload = response.json()

    mapped_status = _map_runpod_status(payload.get("status", "IN_PROGRESS"))
    job["status"] = mapped_status
    if mapped_status == "running":
        job["running_polls"] = int(job.get("running_polls", 0) or 0) + 1
    job["progress_message"] = _compose_progress_message(job, mapped_status, payload)

    if mapped_status == "done":
        audio, duration, error = _extract_runpod_output(payload)
        if error:
            job["status"] = "failed"
            job["error"] = error
            job["progress_message"] = _status_message("failed")
        else:
            job["audio"] = audio
            job["duration"] = duration
            job["completed_at"] = time.time()
            job["progress_message"] = _status_message("done")

    if mapped_status == "failed":
        job["error"] = payload.get("error", "RunPod job failed.")


async def _run_local_tts_job(job_id: str, story: str, tone: str, speaker: str) -> None:
    job = audio_jobs.get(job_id)
    if not job:
        return

    job["status"] = "running"
    job["progress_message"] = _status_message("running")

    try:
        audio_b64, duration = await asyncio.to_thread(generate_audio, story, tone, speaker)
        job["audio"] = f"data:audio/wav;base64,{audio_b64}"
        job["duration"] = duration
        job["status"] = "done"
        job["progress_message"] = _status_message("done")
        job["completed_at"] = time.time()
    except Exception as exc:
        job["status"] = "failed"
        job["progress_message"] = _status_message("failed")
        job["error"] = str(exc)


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


@app.post("/generate-audio-job")
async def create_audio_job(
    story: str = Form(...),
    tone: str = Form("joyful"),
    speaker: str = Form("Lea"),
):
    """Create an async TTS job. Use /generate-audio-job/{job_id} to fetch status."""
    if not story.strip():
        raise HTTPException(status_code=400, detail="Story text is required.")

    job_id = str(uuid.uuid4())
    created_at = time.time()
    job = {
        "job_id": job_id,
        "status": "queued",
        "progress_message": _status_message("queued"),
        "provider": TTS_PROVIDER,
        "provider_job_id": None,
        "expected_chunks": len(split_story(story, max_sentences=2)),
        "running_polls": 0,
        "created_at": created_at,
        "completed_at": None,
        "audio": "",
        "duration": 0.0,
        "error": "",
    }
    audio_jobs[job_id] = job

    try:
        if TTS_PROVIDER == "runpod":
            external_id, status, progress_message = await _submit_runpod_job(story, tone, speaker)
            job["provider_job_id"] = external_id
            job["status"] = status
            job["progress_message"] = progress_message
        else:
            asyncio.create_task(_run_local_tts_job(job_id, story, tone, speaker))
    except Exception as exc:
        job["status"] = "failed"
        job["progress_message"] = _status_message("failed")
        job["error"] = str(exc)

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "provider": job["provider"],
        "progress_message": job["progress_message"],
        "created_at": job["created_at"],
        "expected_chunks": job["expected_chunks"],
    }


@app.get("/generate-audio-job/{job_id}")
async def get_audio_job(job_id: str):
    """Check async TTS job status and return audio when available."""
    job = audio_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Audio job not found.")

    is_final = job["status"] in {"done", "failed"}
    timed_out = (time.time() - float(job["created_at"])) > AUDIO_JOB_TIMEOUT_SEC

    if not is_final and timed_out:
        job["status"] = "failed"
        job["progress_message"] = _status_message("failed")
        job["error"] = "Audio job timed out."

    if job["provider"] == "runpod" and job["status"] not in {"done", "failed"}:
        try:
            await _refresh_runpod_job(job)
        except Exception as exc:
            job["status"] = "failed"
            job["progress_message"] = _status_message("failed")
            job["error"] = f"RunPod status check failed: {exc}"

    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "provider": job["provider"],
        "progress_message": job["progress_message"],
        "expected_chunks": job["expected_chunks"],
        "running_polls": job["running_polls"],
        "audio": job["audio"],
        "duration": job["duration"],
        "error": job["error"],
        "created_at": job["created_at"],
        "completed_at": job["completed_at"],
        "poll_after_sec": RUNPOD_POLL_INTERVAL_SEC,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))