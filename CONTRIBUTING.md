# Contributing to MagicNarrate

Welcome! This guide will help you set up a development environment and contribute to the project.

## Prerequisites

- Node.js 18+
- Python 3.10+
- OpenAI API key
- Optionally: Docker (for containerized testing)

## Setup

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/MagicNarrate.git
cd MagicNarrate
npm install --legacy-peer-deps
```

### 2. Frontend Setup

Copy the frontend environment file:
```bash
cp .env.example .env.local
# Edit .env.local if needed (defaults work for localhost)
```

Run the dev server:
```bash
npm run dev
```

Frontend will be available at `http://localhost:5173`.

### 3. Backend Setup

Create and activate a Python virtual environment:
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy the backend environment file:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

Run the backend server:
```bash
python server.py
```

Backend will be available at `http://localhost:8000`.

## Development Workflow

### Running Tests

```bash
# Frontend tests
npm run test

# Frontend tests in watch mode
npm run test:watch

# Backend tests (with mocked dependencies)
source backend/.venv/bin/activate
python3 -m pytest backend/test_server.py -v
```

### Linting & Formatting

```bash
# Frontend
npm run lint       # Check for lint issues
npm run lint:fix   # Auto-fix lint issues
npm run format     # Format code with Prettier

# Backend
source backend/.venv/bin/activate
ruff check backend  # Check for lint issues
```

### Type Checking

```bash
npm run typecheck
```

### Building

```bash
npm run build      # Frontend production build
```

## Code Standards

- **Frontend:** TypeScript + React with ESLint + Prettier
- **Backend:** Python 3.10+ with ruff linting
- **Format:** Run `npm run format` before committing
- **Tests:** All PRs must have passing tests

## Pre-commit Hooks

Husky and lint-staged are configured to auto-format and lint staged files before each commit.

If you need to bypass (not recommended):
```bash
git commit --no-verify
```

## Git Workflow

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes and run tests: `npm run test`
3. Commit with a clear message
4. Push and create a PR

## Docker

### Building the Backend Container

```bash
# HF Space (FastAPI on port 7860)
docker build -f hf-space/Dockerfile -t magicnarrate-hf hf-space/

# RunPod Worker
docker build -f runpod-worker/Dockerfile -t magicnarrate-runpod runpod-worker/
```

### Running Backend in Docker

```bash
docker run \
  -e OPENAI_API_KEY=your_key \
  -e TTS_PROVIDER=local \
  -p 7860:7860 \
  magicnarrate-hf
```

## Troubleshooting

### `npm install` fails with peer dependency errors

Use `--legacy-peer-deps`:
```bash
npm install --legacy-peer-deps
```

### Backend won't start (missing models)

Make sure model files are in `backend/image_captioning/`:
- `resnet50_attention_model.pth`
- `vocab.pt`

Download from: https://drive.google.com/drive/folders/1H-rOG2pVmDHQMoA9pm8ZqDS1endTIYuV?usp=sharing

### Tests fail in CI but pass locally

Check your environment variables and Python version:
```bash
python3 --version  # Should be 3.10+
echo $OPENAI_API_KEY  # Should be set for backend tests
```

## Questions?

Check the main [README.md](README.md) for architecture and deployment info.
