# MagicNarrate — System Architecture

## Overview

MagicNarrate is an AI-powered storytelling platform that transforms images or text prompts into narrative stories with AI-generated narration. The system consists of a frontend React application and a backend API with machine learning inference capabilities.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Browser (Web Client)                   │
│                       React + TypeScript                         │
│         (LandingPage, SystemPage, StoryResultSection)           │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/REST
                    ┌────────▼────────┐
                    │   API Gateway   │
                    │ (Vercel Frontend)
                    └────────┬────────┘
                             │ HTTP/REST
                ┌────────────▼────────────┐
                │   Backend API Server    │
                │     (FastAPI + Uvicorn) │
                │   Deployment Options:   │
                │  - HF Spaces (free)     │
                │  - RunPod (async jobs)  │
                └────────────┬────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   ┌─────────┐         ┌──────────┐        ┌──────────┐
   │   LLM   │         │   Image  │        │   TTS    │
   │ OpenAI  │         │  Vision  │        │ Parler   │
   │ (GPT)   │         │ ResNet50 │        │ TTS      │
   │         │         │ +        │        │          │
   │ (Story) │         │Attention │        │(Narrate) │
   └─────────┘         └──────────┘        └──────────┘
```

---

## Component Architecture

### Frontend (`/src`)

**Technology Stack:**
- **Framework:** React 18.3.1 with TypeScript 5.5.4 (strict mode)
- **Build Tool:** Vite 6.4.1 (fast dev server, optimized builds)
- **Styling:** Tailwind CSS 3.4.1
- **API Client:** Supabase (can be extended)
- **Analytics:** Vercel Analytics

**Components:**
- **LandingPage**: Main entry point, image/text input interface
- **SystemPage**: Configuration and settings panel
- **StoryResultSection**: Display generated stories and narration controls
- **App.tsx**: Root component with routing logic

**Key Features:**
- Lazy-loaded routes for optimal performance
- Responsive Tailwind design
- Integration with backend API for story/narration generation
- Voice selection for narration playback

### Backend (`/backend`)

**Technology Stack:**
- **Framework:** FastAPI 0.133.1 (async Python web framework)
- **Server:** Uvicorn 0.41.0 (ASGI application server)
- **ML Inference:** PyTorch 2.10.0, Transformers 4.46.1
- **LLM Integration:** OpenAI API 2.24.0 (GPT models for story generation)
- **Text-to-Speech:** Parler TTS 1.3.1 (narration generation)
- **Data Science:** NumPy 2.4.2
- **Config Management:** python-dotenv 1.2.1

**Core Files:**
- **server.py**: Main FastAPI application, endpoints for story/narration generation
- **model_def.py**: Model architecture definitions (ResNet50 + Attention mechanism)

**Deployment Variants:**
- **HF Spaces** (`/hf-space`): Free Hugging Face deployment, includes Dockerfile, requirements.txt
- **RunPod** (`/runpod-worker`): GPU-accelerated serverless, async job handler pattern, includes Dockerfile, handler.py

### Models (`/notebooks`, `/backend/image_captioning`)

**Image Captioning Model:**
- **Architecture:** ResNet50 + Attention Mechanism
- **Input:** Image (preprocessed with ResNet50 features)
- **Output:** Image caption/description
- **Files:**
  - `resnet50_features.pt`: Pre-extracted ResNet50 features
  - `resnet50_attention_model.pth`: Trained attention model weights
  - `vocab.pt`: Vocabulary for caption generation

**Evaluation Notebooks:**
- Multiple training notebooks for model experimentation
- Beam search implementation for caption generation
- BLEU score evaluation metrics

---

## Data Flow

### Story Generation Flow

```
1. User Input (Image or Text)
   ↓
2. Frontend (React)
   - Validate input
   - Call backend API
   ↓
3. Backend (FastAPI)
   - Extract image features (if image input)
   - Generate caption (ResNet50 + Attention)
   - Call OpenAI API with caption + tone/emotion parameters
   - Receive story text from GPT
   ↓
4. Response to Frontend
   - Return story JSON
   - Display story on StoryResultSection
```

### Narration Generation Flow

```
1. Story Text (from previous flow)
   ↓
2. User Selects Voice & Settings
   ↓
3. Backend (FastAPI)
   - Normalize text (handle punctuation, abbreviations)
   - Call Parler TTS with selected voice parameters
   - Generate audio file
   - Return audio URL or stream
   ↓
4. Frontend (React)
   - Render audio player
   - Show voice selection UI
   - Stream or download narration
```

### Asynchronous Job Pattern (RunPod)

For GPU-intensive tasks, RunPod supports async jobs:

```
1. Frontend → Backend
   POST /generate-story
   { image: ..., tone: "..." }
   ↓
2. Backend → RunPod Handler
   - Job ID returned immediately
   - Backend queues job on RunPod
   ↓
3. Frontend Polls
   GET /job-status/{job_id}
   (Poll every ~2-5 seconds)
   ↓
4. RunPod Handler Completes
   - Processes image + LLM
   - Stores result in backend
   ↓
5. Frontend Receives Result
   - Display story + narration
```

---

## Deployment Architecture

### Frontend Deployment (Vercel)

- **Static Site Hosting:** Vercel Edge Network
- **Build:** `npm run build` → Vite output to `/dist`
- **Environment:** VITE_API_URL points to backend
- **Features:** Auto-deployment on git push, preview URLs, analytics

### Backend Deployment Options

#### Option 1: Hugging Face Spaces (Free, CPU)
```
- Dockerfile-based deployment
- Public endpoint
- Limited compute (CPU only)
- Best for: Demo, development, low-traffic
```

#### Option 2: RunPod (GPU, Serverless)
```
- Custom handler pattern (handler.py)
- GPU acceleration (NVIDIA)
- Async job queue support
- Best for: Production, image processing, model inference
- Requires: RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID
```

#### Environment Variables

**Frontend (.env.local):**
```
VITE_API_URL=https://backend-api.example.com
VITE_AUDIO_JOB_POLL_INTERVAL_MS=2000
VITE_AUDIO_JOB_TIMEOUT_MS=300000
```

**Backend (.env):**
```
OPENAI_API_KEY=sk-...
TTS_PROVIDER=parler  # or tts-1, tts-1-hd
RUNPOD_API_KEY=...
RUNPOD_ENDPOINT_ID=...
AUDIO_JOB_POLL_INTERVAL_SEC=2
AUDIO_JOB_TIMEOUT_SEC=300
TTS_PROMPT_MAX_TOKENS=200
```

---

## Development Workflow

### Local Setup

**Frontend:**
```bash
npm install
npm run dev          # Start dev server with hot reload
npm run typecheck    # TypeScript validation
npm run lint         # ESLint checks
npm run test         # Vitest unit tests
npm run build        # Production build
```

**Backend:**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python server.py  # Start FastAPI server
```

### Testing

**Frontend Tests (Vitest + React Testing Library):**
```bash
npm run test          # Run all tests
npm run test:watch   # Watch mode
```

**Backend Tests (pytest):**
```bash
cd backend
pytest test_server.py -v
```

### CI/CD Pipeline (GitHub Actions)

Located in `.github/workflows/ci.yml`:

1. **Frontend Job:**
   - Node 18 environment
   - Dependency install (`npm ci`)
   - Prettier check (formatting)
   - ESLint (linting)
   - TypeScript compilation (type-checking)
   - Vitest (unit tests)
   - Production build

2. **Backend Job:**
   - Python 3.10 environment
   - Install minimal CI deps (`requirements-ci.txt`: pytest, ruff)
   - Ruff linter check (warnings)
   - Python syntax validation (`compileall`)
   - pytest execution

3. **Triggers:**
   - Push to `main` or `master`
   - Pull requests to `main` or `master`

---

## Security & Environment

### Secrets Management

- **OpenAI API Key:** Environment variable only, never committed
- **RunPod API Key:** Environment variable only, never committed
- **.env files:** All local `.env` files in `.gitignore`
- **.env.example:** Template provided for setup

### Pre-commit Hooks (Husky + lint-staged)

- Automatic ESLint on staged files
- Automatic Prettier formatting
- Prevents commits with lint errors

### Dependency Versioning

- **Frontend:** Exact versions in `package-lock.json` for reproducibility
- **Backend:** Pinned versions in `requirements.txt` for reproducibility
- **CI-only deps:** Minimal `requirements-ci.txt` for fast CI builds

---

## Performance Considerations

### Frontend Optimization

- Lazy-loaded route components (reduces initial bundle)
- Tailwind CSS purging (unused styles removed)
- Vite optimizations (pre-bundling, splitting)
- Tree-shaking (dead code elimination)

**Current Metrics:**
- Main bundle: 141.61 KB (gzipped: 45.44 KB)
- Dev server startup: <1s

### Backend Optimization

- Async FastAPI handlers (non-blocking I/O)
- Model inference caching (optional)
- RunPod GPU acceleration for heavy tasks
- Async polling pattern for long-running jobs

---

## Future Enhancements

1. **Caching Layer:** Redis for story/narration caching
2. **Database:** PostgreSQL for user history, favorites, sharing
3. **Authentication:** JWT + OAuth for user accounts
4. **Monitoring:** Sentry for error tracking, Datadog for metrics
5. **E2E Testing:** Playwright for full workflow validation
6. **Load Testing:** k6 for performance benchmarking
7. **API Documentation:** OpenAPI/Swagger auto-generated from FastAPI
8. **Multi-language Support:** i18n for UI + multilingual story generation

---

## References

- **Frontend Docs:** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Deployment Guide:** [DEPLOYMENT.md](../DEPLOYMENT.md)
- **Project Audit:** [AUDIT.md](AUDIT.md)
- **Quick Start:** [DEPLOY_QUICK_START.md](../DEPLOY_QUICK_START.md)
