# Quick Deployment Checklist

## Pre-Deployment

- [ ] All code is committed and pushed to GitHub
- [ ] API keys are ready:
  - [ ] OpenAI API key
  - [ ] HuggingFace token (optional)
- [ ] Backend models are present in `hf-space/image_captioning/`
  - [ ] `resnet50_attention_model.pth`
  - [ ] `vocab.pt`

## Backend Deployment (HuggingFace Spaces)

### Create Space
- [ ] Created new Space on HuggingFace (Docker SDK)
- [ ] Space name: `MagicNarrate`
- [ ] Repository connected to GitHub

### Configure Environment
In HF Space Settings → "Repository secrets":
- [ ] `OPENAI_API_KEY` set
- [ ] `HF_TOKEN` set (if needed)

### Verify Deployment
- [ ] Space shows "Running" status
- [ ] No errors in build logs
- [ ] Test endpoint: `curl https://your-username-MagicNarrate.hf.space/speakers`
- [ ] Response shows available speakers
- [ ] Note Space URL for next step

## Frontend Deployment (Vercel)

### Create Project
- [ ] Connected GitHub repository to Vercel
- [ ] Selected root directory: `./`
- [ ] Build command: `npm run build`
- [ ] Output directory: `dist`

### Configure Environment
In Vercel Project Settings → "Environment Variables":
- [ ] Name: `VITE_API_URL`
- [ ] Value: `https://your-username-MagicNarrate.hf.space`
- [ ] Applied to: Production, Preview, Development

### Verify Deployment
- [ ] Vercel shows "Ready" status
- [ ] Deployment URL is live (e.g., https://magicnarrate.vercel.app)
- [ ] Open URL in browser - no errors
- [ ] Custom domain configured (optional)

## Post-Deployment Testing

### Basic Functionality
- [ ] Frontend loads without errors
- [ ] Can select text input mode
- [ ] Can select image input mode
- [ ] Genre and tone dropdowns work
- [ ] Speaker selection works

### Text Input Test
- [ ] Enter a prompt
- [ ] Click "Generate Story"
- [ ] Story text appears within 30 seconds
- [ ] Audio generation starts
- [ ] Audio plays successfully

### Image Upload Test
- [ ] Click image upload button
- [ ] Select an image file
- [ ] Image preview appears
- [ ] Click "Generate Story"
- [ ] Story text appears within 30 seconds
- [ ] Audio generation and playback work

### Audio Features
- [ ] Can play/pause audio
- [ ] Can skip forward/backward
- [ ] Can download audio file
- [ ] Downloaded file is a valid WAV

## Monitoring & Support

- [ ] Added monitoring alerts (optional)
- [ ] Documented support contact
- [ ] Shared deployment guide with team
- [ ] Environment variables documented securely

## Troubleshooting Resources

If issues occur:
1. Check [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed troubleshooting
2. Review HuggingFace Space build logs
3. Review Vercel deployment logs
4. Check browser console (F12) for CORS errors
5. Verify API keys are correct and active

---

**Deployment Date**: ____________________
**Deployer**: ____________________
**Notes**: _________________________________

