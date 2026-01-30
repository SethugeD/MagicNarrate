# MagicNarrate 🪄📖

An AI-powered storytelling app for kids that generates engaging stories with emotional narration from images or text prompts.

## Features

- 🖼️ **Image-to-Story**: Upload an image and get a creative story based on it
- ✍️ **Text-to-Story**: Enter a prompt and generate a story
- 🎭 **Emotion Tones**: Choose from Joyful, Funny, Mysterious, Calm, Sad, Confused, or Dramatic
- 📚 **Story Genres**: Adventure, Fantasy, Bedtime, Friendship, Learning, Confidence
- 🔊 **Text-to-Speech**: Stories are narrated with emotional voices using Parler-TTS

## Tech Stack

### Frontend
- React + TypeScript
- Vite
- Tailwind CSS

### Backend
- FastAPI (Python)
- PyTorch (Image Captioning)
- Google Gemini API (Story Generation)
- Parler-TTS (Text-to-Speech)

## Setup

### Prerequisites
- Node.js 18+
- Python 3.10+
- Google Gemini API key

### Frontend Setup
```bash
npm install
npm run dev
```

### Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file with your API key
echo "GEMINI_API_KEY=your_api_key_here" > .env

# Download model files (see Model Files section)

# Run server
python server.py
```

### Model Files

The following model files are required but not included in the repo (too large):

Place these in `backend/image_captioning/`:
- `resnet50_model.pth` (~19MB) - Trained caption model
- `resnet50_features.pt` (~66MB) - ResNet50 features
- `vocab.pt` (~80KB) - Vocabulary file

Contact the repository owner for access to these files.

## Usage

1. Start the backend server (port 8000)
2. Start the frontend dev server (port 5173)
3. Open http://localhost:5173
4. Choose text or image input mode
5. Select genre and emotion tone
6. Click "Generate Story" and enjoy!

## API Endpoints

- `POST /generate` - Generate story from image
- `POST /generate-from-text` - Generate story from text prompt
- `GET /speakers` - Get available TTS voices
