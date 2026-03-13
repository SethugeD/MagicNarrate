#!/bin/bash

# Deployment verification script for MagicNarrate
# Usage: ./verify-deployment.sh <hf_space_url> <vercel_url>

set -e

HF_SPACE_URL="${1:-https://huggingface.co/spaces/your-username/MagicNarrate}"
VERCEL_URL="${2:-https://your-project.vercel.app}"

echo "🚀 MagicNarrate Deployment Verification"
echo "========================================"
echo ""

# Extract base URL from HF Space URL (remove trailing spaces path)
HF_API_URL="${HF_SPACE_URL%/spaces/*}"
if [[ $HF_SPACE_URL == *"/spaces/"* ]]; then
  HF_USERNAME=$(echo $HF_SPACE_URL | grep -oP 'spaces/\K[^/]+')
  HF_SPACE_NAME=$(echo $HF_SPACE_URL | grep -oP 'spaces/[^/]+/\K[^/]+')
  HF_API_URL="https://${HF_USERNAME}-${HF_SPACE_NAME}.hf.space"
fi

echo "Checking Backend (HuggingFace Spaces)..."
echo "URL: $HF_API_URL"

# Check speakers endpoint
if curl -s "$HF_API_URL/speakers" > /dev/null 2>&1; then
  echo "✅ Backend is accessible"
  
  SPEAKERS=$(curl -s "$HF_API_URL/speakers" | jq -r '.speakers[]' 2>/dev/null || echo "")
  if [ -n "$SPEAKERS" ]; then
    echo "✅ Available speakers: $SPEAKERS"
  fi
else
  echo "❌ Backend is not responding"
  echo "   Make sure the HF Space is running and has the correct URL"
fi

echo ""
echo "Checking Frontend (Vercel)..."
echo "URL: $VERCEL_URL"

# Check if frontend is accessible
if curl -s -I "$VERCEL_URL" | grep -q "200\|301\|302"; then
  echo "✅ Frontend is accessible"
else
  echo "❌ Frontend is not responding"
  echo "   Make sure Vercel deployment is complete"
fi

echo ""
echo "========================================"
echo "Deployment Verification Complete!"
echo ""
echo "Next steps:"
echo "1. Visit $VERCEL_URL in your browser"
echo "2. Make sure VITE_API_URL environment variable is set to: $HF_API_URL"
echo "3. Test uploading an image or entering text to generate a story"
