# Deployment Quick Start

Estimated time: 10-20 minutes

## Step 1: Confirm RunPod endpoint

Make sure your queue-based endpoint exists and has:

- Endpoint name: `MagicNarrate-tts`
- Dockerfile path: `/runpod-worker/Dockerfile`
- Build context: `/runpod-worker`

Keep these values:

- `RUNPOD_API_KEY`
- `RUNPOD_ENDPOINT_ID`

## Step 2: Configure HF Space secrets

Set these in HF Space Settings -> Secrets:

```env
OPENAI_API_KEY=...
TTS_PROVIDER=runpod
RUNPOD_API_KEY=...
RUNPOD_ENDPOINT_ID=...
AUDIO_JOB_POLL_INTERVAL_SEC=2.5
AUDIO_JOB_TIMEOUT_SEC=420
TTS_PROMPT_MAX_TOKENS=220
```

Redeploy HF Space.

## Step 3: Configure Vercel env vars

Set these in Vercel Project Settings -> Environment Variables:

```env
VITE_API_URL=https://<your-space>.hf.space
VITE_AUDIO_JOB_POLL_INTERVAL_MS=2500
VITE_AUDIO_JOB_TIMEOUT_MS=420000
```

Redeploy Vercel.

## Step 4: Verify backend

```bash
curl -s https://<your-space>.hf.space/health
curl -s -X POST https://<your-space>.hf.space/generate-audio-job \
  -F "story=A tiny fox found a glowing lantern and smiled." \
  -F "tone=joyful" \
  -F "speaker=Lea"
```

If second command returns `job_id`, backend is correctly wired to RunPod.

## Step 5: Verify full app

Open Vercel URL and test:

- Text -> story generation
- Image -> story generation
- Audio status progression and playback

## Optional script

```bash
chmod +x verify-deployment.sh
./verify-deployment.sh https://<your-space>.hf.space https://<your-app>.vercel.app
```
