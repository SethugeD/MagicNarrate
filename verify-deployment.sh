#!/bin/bash

# Deployment verification script for MagicNarrate
# Usage:
#   ./verify-deployment.sh <hf_space_url_or_hf_space_api_url> <vercel_url>

set -e

HF_SPACE_URL="${1:-https://huggingface.co/spaces/your-username/MagicNarrate}"
VERCEL_URL="${2:-https://your-project.vercel.app}"

echo "MagicNarrate Deployment Verification"
echo "===================================="
echo

HF_API_URL="$HF_SPACE_URL"
if [[ $HF_SPACE_URL == *"/spaces/"* ]]; then
  HF_USERNAME=$(echo "$HF_SPACE_URL" | grep -oE 'spaces/[^/]+' | cut -d/ -f2)
  HF_SPACE_NAME=$(echo "$HF_SPACE_URL" | grep -oE 'spaces/[^/]+/[^/]+' | cut -d/ -f3)
  HF_API_URL="https://${HF_USERNAME}-${HF_SPACE_NAME}.hf.space"
fi

echo "Backend URL: $HF_API_URL"

health_payload=$(curl -s "$HF_API_URL/health" || true)
if echo "$health_payload" | grep -q '"status":"healthy"'; then
  echo "PASS: /health"
else
  echo "FAIL: /health"
  echo "Response: $health_payload"
  exit 1
fi

root_payload=$(curl -s "$HF_API_URL/" || true)
if echo "$root_payload" | grep -q 'generate-audio-job'; then
  echo "PASS: async backend version deployed"
else
  echo "WARN: root message does not mention generate-audio-job"
  echo "Response: $root_payload"
fi

speakers_payload=$(curl -s "$HF_API_URL/speakers" || true)
if echo "$speakers_payload" | grep -q '"speakers"'; then
  echo "PASS: /speakers"
else
  echo "FAIL: /speakers"
  echo "Response: $speakers_payload"
  exit 1
fi

echo
echo "Checking async TTS job create endpoint..."
job_payload=$(curl -s -X POST "$HF_API_URL/generate-audio-job" \
  -F "story=A tiny fox found a glowing lantern and smiled." \
  -F "tone=joyful" \
  -F "speaker=Lea" || true)

if echo "$job_payload" | grep -q '"job_id"'; then
  job_id=$(echo "$job_payload" | sed -n 's/.*"job_id":"\([^"]*\)".*/\1/p')
  echo "PASS: /generate-audio-job created job_id=$job_id"
else
  echo "FAIL: /generate-audio-job"
  echo "Response: $job_payload"
  exit 1
fi

echo
echo "Frontend URL: $VERCEL_URL"
if curl -s -I "$VERCEL_URL" | grep -q "200\|301\|302"; then
  echo "PASS: frontend reachable"
else
  echo "FAIL: frontend not reachable"
  exit 1
fi

echo
echo "Verification complete."
echo "Next: open $VERCEL_URL and run one full story + audio test in browser."
