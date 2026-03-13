# MagicNarrate 🪄

MagicNarrate is an AI-powered storytelling app that turns image or text prompts into children's stories, then narrates them with selectable voices.

## Live Architecture

- Frontend: Vercel (React + Vite)
- Backend: HuggingFace Spaces (FastAPI + Docker)
- Story generation: OpenAI API
- Voice narration: Parler-TTS

## Features

- Image-to-story generation
- Text-to-story generation
- Genre and emotion-tone controls
- Multiple speaker voices (Jon, Lea, Gary, Jenna)
- Audio playback controls and WAV download

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
```

For production (Vercel), set `VITE_API_URL` to your Space URL:

```env
VITE_API_URL=https://damz25-magic-narrate.hf.space
```

### Backend (`backend/.env` locally, Space Secrets in production)

```env
OPENAI_API_KEY=your_openai_api_key
```

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
- `POST /generate-audio` generate narrated audio from story text using selected speaker

## Deployment

- Frontend deployment is configured for Vercel using `vercel.json`
- Backend deployment is configured for HuggingFace Spaces using `hf-space/Dockerfile`
- Set environment variables in each platform dashboard:
	- Vercel: `VITE_API_URL`
	- HuggingFace Space Secrets: `OPENAI_API_KEY`
