# MagicNarrate 🪄

MagicNarrate is an AI-powered storytelling app that turns image or text prompts into children's stories, then narrates them with selectable voices.

## Live Architecture

- Frontend: Vercel (React + Vite)
- Backend: HuggingFace Spaces (FastAPI + Docker)
- Audio Inference: RunPod Serverless (optional, recommended for lower TTS latency)
- Story generation: OpenAI API
- Voice narration: Parler-TTS

## Features

- Image-to-story generation
- Text-to-story generation
- Genre and emotion-tone controls
- Multiple speaker voices (Jon, Lea, Gary, Jenna)
- Audio playback with seek, skip, replay controls and WAV download
- Editable story text with one-click audio regeneration

## Tech Stack

### Frontend
- React
- TypeScript
- Vite
- Tailwind CSS

### Backend
- FastAPI
- PyTorch
- torchvision
- OpenAI Python SDK
- Parler-TTS

## Project Structure

- `src/` frontend source code
- `backend/` local backend for development
- `hf-space/` deployed backend source for HuggingFace Space

## Local Development

### Prerequisites

- Node.js 18+
- Python 3.10+
- OpenAI API key

### 1) Start backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=your_api_key_here" > .env
python server.py
```

Backend runs at `http://localhost:8000`.

### 2) Start frontend

```bash
cd ..
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`.

## Environment Variables

### Frontend (`.env.local`)

```env
VITE_API_URL=http://localhost:8000
VITE_AUDIO_JOB_POLL_INTERVAL_MS=2500
VITE_AUDIO_JOB_TIMEOUT_MS=420000
```

For production (Vercel), set `VITE_API_URL` to your Space URL:

```env
VITE_API_URL=https://damz25-magic-narrate.hf.space
```

### Backend (`backend/.env` locally, Space Secrets in production)

```env
OPENAI_API_KEY=your_openai_api_key
TTS_PROVIDER=local
# When using RunPod, set TTS_PROVIDER=runpod and configure:
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_ID=your_runpod_endpoint_id
AUDIO_JOB_POLL_INTERVAL_SEC=2.5
AUDIO_JOB_TIMEOUT_SEC=420
TTS_PROMPT_MAX_TOKENS=220
```

`TTS_PROVIDER` behavior:
- `local`: uses Parler-TTS directly in this backend process.
- `runpod`: creates async jobs in RunPod Serverless and polls status until complete.

## Model Files

Required files for backend inference:

For `backend/image_captioning/`:
- `resnet50_attention_model.pth`
- `resnet50_features.pt`
- `vocab.pt`

For `hf-space/image_captioning/`:
- `resnet50_attention_model.pth`
- `vocab.pt`

## API Endpoints

- `GET /` basic service info
- `GET /health` health check
- `GET /speakers` available speaker list
- `POST /generate` generate story from uploaded image
- `POST /generate-from-text` generate story from text prompt
- `POST /generate-audio` synchronous local audio generation (legacy/fallback)
- `POST /generate-audio-job` create async audio job
- `GET /generate-audio-job/{job_id}` check audio job status and fetch result when done

## RunPod Serverless Setup

1. Create a RunPod account and generate an API key.
2. Push this repository branch with the `runpod-worker/` folder to GitHub.
3. Create a Serverless endpoint from GitHub with these values:

- Repo: `SethugeD/MagicNarrate`
- Branch: the branch containing `runpod-worker/`
- Dockerfile Path: `/runpod-worker/Dockerfile`
- Build Context: `/runpod-worker`
- Endpoint Type: queue-based
- Endpoint Name: `MagicNarrate-tts`

4. If RunPod asks for handler validation, it should detect `runpod.serverless.start({"handler": handler})` from `runpod-worker/handler.py`.
5. Make sure the worker accepts this input JSON shape:

```json
{
	"input": {
		"story": "string",
		"tone": "joyful",
		"speaker": "Lea"
	}
}
```

6. Make sure your worker output includes at least one of these fields:
- `audio` (base64 or data URL)
- `audio_base64`
- `wav_base64`
- `audio_url`

7. Set backend env variables (`backend/.env` locally, HF Space Secrets in production):
- `TTS_PROVIDER=runpod`
- `RUNPOD_API_KEY=<your key>`
- `RUNPOD_ENDPOINT_ID=<your endpoint id>`

8. Restart backend and verify flow:
- `POST /generate-audio-job`
- `GET /generate-audio-job/{job_id}` until status is `done`

## RunPod Notes

- For occasional traffic, Serverless is the best default due to low idle cost.
- If cold starts become painful, migrate only the TTS worker to a dedicated RunPod pod during active demo windows.

## Deployment

- Frontend deployment is configured for Vercel using `vercel.json`
- Backend deployment is configured for HuggingFace Spaces using `hf-space/Dockerfile`
- Set environment variables in each platform dashboard:
	- Vercel: `VITE_API_URL`, `VITE_AUDIO_JOB_POLL_INTERVAL_MS`, `VITE_AUDIO_JOB_TIMEOUT_MS`
	- HuggingFace Space Secrets: `OPENAI_API_KEY`, `TTS_PROVIDER`, `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`, `AUDIO_JOB_POLL_INTERVAL_SEC`, `AUDIO_JOB_TIMEOUT_SEC`
	- Optional TTS stability tuning: `TTS_PROMPT_MAX_TOKENS` (default `220`)
