# Deployment Quick Start

This guide provides step-by-step instructions to deploy MagicNarrate to production.

> **⏱️ Estimated Time**: 20-30 minutes
> **📋 Prerequisites**: OpenAI API key, HuggingFace account, Vercel account

## 🚀 5-Step Deployment

### Step 1: Push Code to GitHub
```bash
git add .
git commit -m "Deploy: Add Vercel and HF Space configuration"
git push origin main
```

### Step 2: Deploy Backend (5 min)

1. Go to https://huggingface.co/spaces and click "Create new Space"
2. Configuration:
   - Space name: `MagicNarrate`
   - SDK: Docker
   - Hardware: CPU basic (or GPU for faster inference)
3. In settings, add "Repository secrets":
   - `OPENAI_API_KEY`: [your OpenAI API key]
   - `HF_TOKEN`: [your HF token - optional]
4. Wait for build to complete (5-10 min)
5. Note your Space URL: `https://[username]-MagicNarrate.hf.space`

### Step 3: Deploy Frontend (5 min)

1. Go to https://vercel.com/new
2. Import your GitHub repository
3. Configure:
   - Framework: Vite (auto-detected)
   - Build: `npm run build`
   - Output: `dist`
4. Click "Deploy"

### Step 4: Configure Environment Variable (2 min)

In Vercel Dashboard:
1. Go to Project Settings → Environment Variables
2. Add new variable:
   - Name: `VITE_API_URL`
   - Value: `https://[username]-MagicNarrate.hf.space`
   - Environments: Production, Preview, Development
3. Click "Add"

### Step 5: Redeploy Frontend (2 min)

1. Go to Vercel Deployments tab
2. Find latest deployment
3. Click "Redeploy" (or push a new commit to trigger auto-deploy)
4. Wait for build to complete

## ✅ Verification

Once deployed, test your app:

```bash
# From terminal (optional - run this script)
chmod +x verify-deployment.sh
./verify-deployment.sh https://huggingface.co/spaces/[username]/MagicNarrate https://[project].vercel.app
```

Or manually:
1. Open Vercel URL in browser
2. Try text input: "Write a magical story about a wizard"
3. Try image upload: Select an image file
4. Verify audio generation and playback work

## 🔍 Where to Find URLs

**Backend URL** (HuggingFace Spaces):
- After Space creation, copy: `https://[username]-MagicNarrate.hf.space`

**Frontend URL** (Vercel):
- After deployment, copy: `https://[project-name].vercel.app`
- Or use custom domain if configured

## 🆘 Need Help?

See detailed troubleshooting in [DEPLOYMENT.md](./DEPLOYMENT.md)

**Common Issues**:
- ❌ "Failed to generate story" → Check `VITE_API_URL` in Vercel settings
- ❌ Backend not responding → Check HF Space build logs
- ❌ CORS errors → Verify Space CORS middleware allows all origins

## 📊 Estimated Costs

- **HuggingFace**: Free (CPU) or $7.50+/month (GPU)
- **Vercel**: Free (included)
- **OpenAI**: ~$0.001-0.01 per story (pay-as-you-go)

## 🎯 Next (Optional)

- Configure custom domain on Vercel
- Set up GitHub Actions for automated testing
- Add monitoring and error tracking
- Enable analytics on Vercel

---

**Ready to deploy?** Start with [Step 2: Deploy Backend](#step-2-deploy-backend-5-min) 🚀
