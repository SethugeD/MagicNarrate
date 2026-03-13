# Deployment Checklist

## Pre-Deploy

- [ ] Latest code pushed to GitHub
- [ ] RunPod endpoint created and healthy
- [ ] OpenAI API key ready

## RunPod (TTS)

- [ ] Queue-based endpoint created
- [ ] Uses `runpod-worker/` Docker image build
- [ ] Endpoint ID recorded
- [ ] API key recorded
- [ ] Min workers `0`, Max workers `1`
- [ ] Container disk set to at least `20 GB`

## HF Space (Backend)

- [ ] Space build passes
- [ ] Secrets set:
  - [ ] `OPENAI_API_KEY`
  - [ ] `TTS_PROVIDER=runpod`
  - [ ] `RUNPOD_API_KEY`
  - [ ] `RUNPOD_ENDPOINT_ID`
  - [ ] `AUDIO_JOB_POLL_INTERVAL_SEC=2.5`
  - [ ] `AUDIO_JOB_TIMEOUT_SEC=420`
- [ ] `GET /health` returns healthy
- [ ] `POST /generate-audio-job` returns `job_id`

## Vercel (Frontend)

- [ ] Env vars set:
  - [ ] `VITE_API_URL=https://<your-space>.hf.space`
  - [ ] `VITE_AUDIO_JOB_POLL_INTERVAL_MS=2500`
  - [ ] `VITE_AUDIO_JOB_TIMEOUT_MS=420000`
- [ ] Frontend redeployed after env changes
- [ ] Homepage loads with no console API errors

## End-to-End Validation

- [ ] Text story generation works
- [ ] Image story generation works
- [ ] Audio job lifecycle works (`queued` -> `done`)
- [ ] Audio playback works
- [ ] Audio download works

## Final Sanity

- [ ] Run `./verify-deployment.sh <hf_api_url> <vercel_url>`
- [ ] Keep one short warm-up request ready for demo sessions
- [ ] If queue grows, temporarily raise max workers to `2`
