import io
import base64
import warnings

import numpy as np
import soundfile as sf
import torch
import runpod
from transformers import AutoTokenizer, set_seed
from parler_tts import ParlerTTSForConditionalGeneration

MODEL_NAME = "parler-tts/parler-tts-mini-expresso"

TONE_TO_EMOTION = {
    "joyful": "happy",
    "funny": "laughing",
    "mysterious": "whisper",
    "confused": "confused",
    "calm": "default",
    "sad": "sad",
    "dramatic": "emphasis",
}

AVAILABLE_SPEAKERS = ["Jon", "Lea", "Gary", "Jenna"]

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Loading TTS model on {DEVICE}...")
tts_model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL_NAME).to(DEVICE)
tts_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("RunPod worker ready.")


def split_story(text: str, max_sentences: int = 2) -> list[str]:
    chunks = []
    sentence = ""
    count = 0
    for char in text.strip():
        sentence += char
        if char in ".!?":
            count += 1
        if count >= max_sentences:
            chunk = sentence.strip()
            if chunk:
                chunks.append(chunk)
            sentence = ""
            count = 0
    if sentence.strip():
        chunks.append(sentence.strip())
    return chunks


def generate_audio_b64(story_text: str, tone: str, speaker: str) -> tuple[str, float, int]:
    tts_emotion = TONE_TO_EMOTION.get(tone, "default")
    description = f"{speaker} speaks slowly in a {tts_emotion} tone with emphasis and high quality audio."

    desc_tokens = tts_tokenizer(description, return_tensors="pt", truncation=True)
    input_ids = desc_tokens.input_ids.to(DEVICE)
    attention_mask = desc_tokens.attention_mask.to(DEVICE)

    story_chunks = split_story(story_text, max_sentences=2)
    total_chunks = len(story_chunks)
    all_audio = []

    set_seed(42)

    for idx, chunk in enumerate(story_chunks, start=1):
        print(f"[runpod-worker] generating chunk {idx}/{total_chunks}")
        prompt_tokens = tts_tokenizer(chunk, return_tensors="pt", truncation=True)
        prompt_input_ids = prompt_tokens.input_ids.to(DEVICE)
        prompt_attention_mask = prompt_tokens.attention_mask.to(DEVICE)

        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                audio = tts_model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask,
                )

        audio_np = audio.detach().cpu().numpy().squeeze()
        all_audio.append(audio_np)
        pause = np.zeros(int(0.4 * tts_model.config.sampling_rate))
        all_audio.append(pause)

    final_audio = np.concatenate(all_audio)
    buffer = io.BytesIO()
    sf.write(buffer, final_audio, tts_model.config.sampling_rate, format="WAV")
    audio_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    duration = len(final_audio) / tts_model.config.sampling_rate
    return audio_b64, duration, total_chunks


def handler(job):
    job_input = job.get("input", {})
    story = str(job_input.get("story", "")).strip()
    tone = str(job_input.get("tone", "joyful")).strip().lower()
    speaker = str(job_input.get("speaker", "Lea")).strip()

    if not story:
        return {"error": "story is required"}

    if speaker not in AVAILABLE_SPEAKERS:
        speaker = "Lea"

    try:
        audio_b64, duration, chunks_total = generate_audio_b64(story, tone, speaker)
        return {
            "audio_base64": audio_b64,
            "duration": duration,
            "chunks_total": chunks_total,
            "chunks_generated": chunks_total,
            "speaker": speaker,
            "tone": tone,
        }
    except Exception as exc:
        return {"error": str(exc)}


runpod.serverless.start({"handler": handler})
