# MagicNarrate Deployment Guide

This guide covers the current production setup:

- Frontend: Vercel (React + Vite)
- Backend API: HuggingFace Spaces (FastAPI, Docker)
- TTS Inference: RunPod Serverless (queue-based endpoint)

## Architecture

Vercel frontend -> HF Space API -> RunPod TTS worker -> audio returned to frontend

## Prerequisites

- OpenAI API key
- HuggingFace account and Space
- RunPod account with serverless endpoint
- Vercel account
- GitHub repo for source control

## 1) RunPod Setup (TTS)

Create a queue-based serverless endpoint using the worker folder in this repo:

- Repo: `SethugeD/MagicNarrate`
- Branch: your deployment branch
- Dockerfile Path: `/runpod-worker/Dockerfile`
- Build Context: `/runpod-worker`
- Endpoint name: `MagicNarrate-tts`

Recommended initial endpoint settings:

- Min workers: `0`
- Max workers: `1`
- Container disk: `20-30 GB`
- Idle timeout: `10 min`

Record these values:

- RunPod API key
- RunPod endpoint ID

## 2) Backend Setup (HF Space)

Use Docker SDK in HF Space and deploy backend API from `hf-space/` files.

Set these HF Space secrets:

- `OPENAI_API_KEY`
- `TTS_PROVIDER=runpod`
- `RUNPOD_API_KEY`
- `RUNPOD_ENDPOINT_ID`
- `AUDIO_JOB_POLL_INTERVAL_SEC=2.5`
- `AUDIO_JOB_TIMEOUT_SEC=420`

Deploy and verify:

- `GET /health` returns `{"status":"healthy"}`
- `POST /generate-audio-job` returns `job_id`

## 3) Frontend Setup (Vercel)

Set these environment variables in Vercel project settings:

- `VITE_API_URL=https://<your-space>.hf.space`
- `VITE_AUDIO_JOB_POLL_INTERVAL_MS=2500`
- `VITE_AUDIO_JOB_TIMEOUT_MS=420000`

Redeploy frontend after changing env values.

## 4) Verify Production

Use script:

```bash
chmod +x verify-deployment.sh
./verify-deployment.sh https://<your-space>.hf.space https://<your-app>.vercel.app
```

Manual checks:

- Frontend loads without API/CORS errors
- Generate story from text and from image
- Audio job transitions from `queued` to `done`
- Audio playback and download work

## Troubleshooting

### `POST /generate-audio-job` returns `404`

HF Space is running old backend code. Redeploy the updated backend.

### Jobs remain `queued`

RunPod worker likely cold, unavailable, or underprovisioned.

- Check RunPod endpoint logs
- Increase max workers temporarily to `2`
- Warm endpoint with one short test request before demos

### Frontend still calls old API

Vercel envs are build-time values for Vite.

- Confirm `VITE_API_URL`
- Trigger a fresh Vercel redeploy

## Local Development

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

Frontend:

```bash
npm install
npm run dev
```

Use `.env.local` for frontend and `backend/.env` for backend secrets.
