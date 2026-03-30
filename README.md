# MagicNarrate 🪄

MagicNarrate is an emotion-aware AI storytelling app that converts image or text prompts into children's stories, then narrates them with selectable voices.

## Live Architecture

- Frontend: Vercel (React + Vite)
- Backend API: Hugging Face Spaces (FastAPI + Docker)
- Audio Inference: RunPod Serverless (recommended in production)
- Story generation: OpenAI API
- Voice narration: Parler-TTS

## Features

- Image-to-story generation
- Text-to-story generation
- Genre and tone controls
- Multiple speaker voices (Jon, Lea, Gary, Jenna)
- Audio playback with seek, skip, replay, and WAV download
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

## Repository Structure

- `src/`: frontend app
- `backend/`: local development backend
- `hf-space/`: Hugging Face Space backend source
- `runpod-worker/`: RunPod serverless worker
- `models.txt`: model artifact source link and update metadata

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

For production (Vercel):

```env
VITE_API_URL=https://<your-space>.hf.space
```

### Backend (`backend/.env` locally, HF Space Secrets in production)

```env
OPENAI_API_KEY=your_openai_api_key
TTS_PROVIDER=local

# Required when using RunPod
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_ENDPOINT_ID=your_runpod_endpoint_id
AUDIO_JOB_POLL_INTERVAL_SEC=2.5
AUDIO_JOB_TIMEOUT_SEC=420
TTS_PROMPT_MAX_TOKENS=220
```

`TTS_PROVIDER` modes:

- `local`: generate audio in the backend process using Parler-TTS.
- `runpod`: create async RunPod jobs and poll until completion.

## Model Files

Model source:

- Google Drive folder (also tracked in `models.txt`): https://drive.google.com/drive/folders/1H-rOG2pVmDHQMoA9pm8ZqDS1endTIYuV?usp=sharing

Required files for `backend/image_captioning/`:

- `resnet50_attention_model.pth`
- `resnet50_features.pt`
- `vocab.pt`

Required files for `hf-space/image_captioning/`:

- `resnet50_attention_model.pth`
- `vocab.pt`

Quick setup:

1. Download artifacts from the Google Drive folder.
2. Put all three files into `backend/image_captioning/` for local development.
3. Put `resnet50_attention_model.pth` and `vocab.pt` into `hf-space/image_captioning/` for Space deployment.

## Deployment

### Frontend (Vercel)

- Uses `vercel.json` with Vite build output (`dist`).
- Set environment variables:
	- `VITE_API_URL=https://<your-space>.hf.space`
	- `VITE_AUDIO_JOB_POLL_INTERVAL_MS=2500`
	- `VITE_AUDIO_JOB_TIMEOUT_MS=420000`
- Redeploy Vercel after any env variable changes.

### Backend (Hugging Face Spaces)

- Use Docker SDK and deploy from `hf-space/`.
- Docker entrypoint is `hf-space/Dockerfile` (uvicorn on port 7860).
- Set Space secrets:
	- `OPENAI_API_KEY`
	- `TTS_PROVIDER=runpod`
	- `RUNPOD_API_KEY`
	- `RUNPOD_ENDPOINT_ID`
	- `AUDIO_JOB_POLL_INTERVAL_SEC=2.5`
	- `AUDIO_JOB_TIMEOUT_SEC=420`
	- Optional: `TTS_PROMPT_MAX_TOKENS=220`

Important note about this repository:

- `hf-space/` is currently ignored by `.gitignore`.
- If your HF Space deploys directly from this GitHub repo, files in `hf-space/` will not be included unless you change ignore rules or use a separate repo/source for Space files.

### RunPod (TTS worker)

Create a queue-based serverless endpoint with:

- Dockerfile path: `/runpod-worker/Dockerfile`
- Build context: `/runpod-worker`
- Endpoint name: `MagicNarrate-tts`

Recommended starting settings:

- Min workers: `0`
- Max workers: `1`
- Container disk: `20-30 GB`

## Deployment Verification

Run the verification script:

```bash
chmod +x verify-deployment.sh
./verify-deployment.sh https://<your-space>.hf.space https://<your-app>.vercel.app
```

Expected checks:

- `GET /health` is healthy
- `POST /generate-audio-job` returns `job_id`
- frontend URL is reachable

## API Endpoints

- `GET /`: basic service information
- `GET /health`: health check
- `GET /speakers`: available speakers
- `POST /generate`: generate story from image
- `POST /generate-from-text`: generate story from text prompt
- `POST /generate-audio`: synchronous local audio route (legacy/fallback)
- `POST /generate-audio-job`: create async audio job
- `GET /generate-audio-job/{job_id}`: check async audio job status

## Additional Deployment Docs

- `DEPLOYMENT.md`
- `DEPLOY_QUICK_START.md`
- `DEPLOYMENT_CHECKLIST.md`
