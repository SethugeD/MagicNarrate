# MagicNarrate Deployment Guide

This guide covers deploying MagicNarrate to **Vercel** (frontend) and **HuggingFace Spaces** (backend).

## System Architecture

```
┌─────────────────────────────────────┐
│     Vercel (Frontend - React)       │
│     - Hosted at custom domain       │
│     - CDN edge locations worldwide  │
└────────────┬────────────────────────┘
             │
             │ API Calls
             │
             ▼
┌─────────────────────────────────────┐
│  HuggingFace Spaces (Backend)       │
│  - FastAPI server running on Space  │
│  - Model inference (TTS, Captions)  │
│  - OpenAI API integration           │
└─────────────────────────────────────┘
```

---

## Prerequisites

### Required API Keys
1. **OpenAI API Key** - For story generation
   - Sign up at https://platform.openai.com/
   - Create API key in Account > API keys

2. **HuggingFace Token** (Optional but recommended)
   - Sign up at https://huggingface.co/
   - Create token at https://huggingface.co/settings/tokens
   - Needed if using restricted models

### Required Tools
- Git
- Node.js 18+ (for local frontend testing)
- GitHub account (for source code)
- Vercel account (https://vercel.com)
- HuggingFace account (https://huggingface.co)

---

## Part 1: Deploy Backend to HuggingFace Spaces

### Step 1: Create New HuggingFace Space

1. Go to https://huggingface.co/spaces
2. Click "Create new Space"
3. Configuration:
   - **Owner**: Your username
   - **Space name**: `MagicNarrate` (or your preference)
   - **License**: Choose appropriate license
   - **Space SDK**: Select "Docker"
   - **Space hardware**: Select "CPU basic" or higher (GPU recommended for faster inference)
   - **Private/Public**: Choose based on preference

### Step 2: Connect Repository

1. In your local repository, push changes to GitHub (if not already):
   ```bash
   git add .
   git commit -m "Add Vercel and deployment config"
   git push origin main
   ```

2. In HuggingFace Space settings:
   - Connect your GitHub repository
   - Select `hf-space/` as the source folder (or root, depending on setup)

### Step 3: Configure Environment Variables

In HuggingFace Space settings → "Repository secrets":

Add these secrets:
- `OPENAI_API_KEY`: Your OpenAI API key
- `HF_TOKEN`: (Optional) Your HuggingFace token

The Space will automatically read these `.env` values.

### Step 4: Verify Backend Deployment

Once deployed (give it 5-10 minutes to build):

1. Open your HF Space URL (e.g., `https://huggingface.co/spaces/your-username/MagicNarrate`)
2. Test endpoints:
   ```bash
   curl https://your-username-MagicNarrate.hf.space/speakers
   ```

3. Note down your **HF Space URL** - you'll need this for frontend deployment.

---

## Part 2: Deploy Frontend to Vercel

### Step 1: Connect Repository to Vercel

1. Go to https://vercel.com/new
2. Import your GitHub repository
3. Configure project:
   - **Framework**: Vite
   - **Root Directory**: `./` (root)
   - **Build Command**: `npm run build` (should auto-detect)
   - **Output Directory**: `dist` (should auto-detect)

### Step 2: Add Environment Variables

In Vercel project settings → "Environment Variables":

Add:
- **Name**: `VITE_API_URL`
- **Value**: Your HF Space URL (e.g., `https://your-username-MagicNarrate.hf.space`)
- **Environments**: Production, Preview, Development

### Step 3: Configure Custom Domain (Optional)

1. Go to project settings → "Domains"
2. Add your custom domain or use Vercel's default domain

### Step 4: Deploy

1. Push changes to GitHub (if not already pushed):
   ```bash
   git push origin main
   ```

2. Vercel will automatically build and deploy on every push to `main` branch

3. Once deployed, your frontend will be live at the provided URL

---

## Verification Checklist

- [ ] Backend Space is running and responding to requests
- [ ] Frontend URL is accessible
- [ ] Environment variables are set correctly in Vercel
- [ ] Image upload functionality works
- [ ] Text input generates stories
- [ ] Audio generation works with story output
- [ ] Custom domain resolves (if applicable)

---

## Troubleshooting

### Frontend shows "Failed to generate story"

1. Check browser console (F12) for CORS errors
2. Verify `VITE_API_URL` environment variable in Vercel settings
3. Ensure HF Space backend is running:
   ```bash
   curl -I https://your-space-url/speakers
   ```

### Backend Space keeps restarting

1. Check HF Space logs for errors
2. Verify `OPENAI_API_KEY` is set correctly
3. Ensure model files are present in `image_captioning/` folder

### Slow performance

1. Consider upgrading HF Space hardware (GPU option)
2. Check Vercel analytics for slow endpoints
3. Monitor model loading times in backend logs

---

## Local Development (Testing Before Deployment)

### Test Backend Locally

```bash
cd backend
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
python server.py
```

The backend will be available at `http://localhost:8000`

### Test Frontend Locally

```bash
npm install
npm run dev
# Frontend will be at http://localhost:5173
```

The frontend will automatically use `http://localhost:8000` for API calls (from `.env.local`).

---

## Auto-Deployment on Code Changes

Both Vercel (frontend) and HuggingFace Spaces (if connected to GitHub) will automatically redeploy when you push changes to your GitHub repository.

---

## Cost Considerations

### HuggingFace Spaces
- Free tier: Limited compute, restarts if inactive
- Paid tiers: $7.50-$30/month for persistent uptime

### Vercel
- Free tier: Suitable for most projects (performance optimizations included)
- Pro tier: $20/month for advanced features

### For Context
- OpenAI API: Pay-per-use (~$0.001-$0.01 per story depending on model)
- HuggingFace models: Free to use (many open-source models included)

---

## Next Steps

1. Configure API keys in HuggingFace Space
2. Deploy backend to HuggingFace Spaces
3. Add `VITE_API_URL` to Vercel environment
4. Deploy frontend to Vercel
5. Test all features on production URLs
6. Set custom domain (optional)

Need help? Check HuggingFace and Vercel documentation or open an issue on GitHub!
