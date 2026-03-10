import { useState, useRef, useEffect, lazy, Suspense } from 'react';
import { Home, ImagePlus, Type, Wand2, Sparkles } from 'lucide-react';

const StoryResultSection = lazy(() => import('./StoryResultSection'));

interface SystemPageProps {
  onBackToHome: () => void;
}

type InputMode = 'text' | 'image';

function SystemPage({ onBackToHome }: SystemPageProps) {
  const [inputMode, setInputMode] = useState<InputMode>('text');
  const [textInput, setTextInput] = useState('');
  const [imagePreview, setImagePreview] = useState<string>('');
  const [genre, setGenre] = useState('adventure');
  const [emotionTone, setEmotionTone] = useState('calm');
  const [generatedStory, setGeneratedStory] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isGeneratingAudio, setIsGeneratingAudio] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const [audioSrc, setAudioSrc] = useState<string>('');
  const [speakers, setSpeakers] = useState<string[]>(['Jon', 'Lea', 'Gary', 'Jenna']);
  const [selectedSpeaker, setSelectedSpeaker] = useState('Lea');
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const progressIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleGenerateStory = async () => {
    setIsGenerating(true);
    setGeneratedStory('');
    setAudioSrc('');
    setIsGeneratingAudio(false);
    
    try {
      const formData = new FormData();
      formData.append('genre', genre);
      formData.append('tone', emotionTone);
      formData.append('speaker', selectedSpeaker);
      
      let endpoint = 'http://localhost:8000';
      
      if (inputMode === 'text') {
        endpoint += '/generate-from-text';
        formData.append('prompt', textInput);
      } else {
        endpoint += '/generate';
        // Convert base64 image to blob
        if (fileInputRef.current?.files?.[0]) {
          formData.append('image', fileInputRef.current.files[0]);
        } else {
          throw new Error('Please select an image');
        }
      }
      
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Failed to generate story');
      }
      
      const data = await response.json();
      setGeneratedStory(data.story);
      setIsGenerating(false);
      
      setIsGeneratingAudio(true);
      
      const audioFormData = new FormData();
      audioFormData.append('story', data.story);
      audioFormData.append('tone', emotionTone);
      audioFormData.append('speaker', selectedSpeaker);
      
      try {
        const audioResponse = await fetch('http://localhost:8000/generate-audio', {
          method: 'POST',
          body: audioFormData,
        });
        
        if (!audioResponse.ok) {
          throw new Error('Failed to generate audio');
        }
        
        const audioData = await audioResponse.json();
        setAudioSrc(audioData.audio);
        setDuration(audioData.duration);
      } finally {
        setIsGeneratingAudio(false);
      }
      
    } catch (error) {
      console.error('Error generating story:', error);
      alert('Failed to generate story. Make sure the backend is running.');
      setIsGenerating(false);
      setIsGeneratingAudio(false);
    }
  };

  const handleDownloadAudio = async () => {
    if (!audioSrc) return;

    try {
      const response = await fetch(audioSrc);
      const audioBlob = await response.blob();
      const url = URL.createObjectURL(audioBlob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `magicnarrate-${selectedSpeaker.toLowerCase()}-story.wav`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading audio:', error);
      alert('Failed to download audio. Please try again.');
    }
  };

  useEffect(() => {
    return () => {
      if (progressIntervalRef.current) {
        clearInterval(progressIntervalRef.current);
      }
      if (audioRef.current) {
        audioRef.current.pause();
      }
    };
  }, []);

  useEffect(() => {
    const fetchSpeakers = async () => {
      try {
        const response = await fetch('http://localhost:8000/speakers');
        if (!response.ok) return;

        const data = await response.json();
        if (Array.isArray(data.speakers) && data.speakers.length > 0) {
          setSpeakers(data.speakers);
          setSelectedSpeaker((prev) => (data.speakers.includes(prev) ? prev : data.speakers[0]));
        }
      } catch (error) {
      }
    };

    fetchSpeakers();
  }, []);

  useEffect(() => {
    if (audioSrc && audioRef.current) {
      audioRef.current.src = audioSrc;
      audioRef.current.load();
    }
  }, [audioSrc]);

  const handlePlayPause = () => {
    if (!audioRef.current) return;

    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleStop = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    setIsPlaying(false);
    setProgress(0);
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const newProgress = parseFloat(e.target.value);
    setProgress(newProgress);
    if (audioRef.current && duration > 0) {
      audioRef.current.currentTime = newProgress * duration;
    }
  };

  const handleSkipForward = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.min(audioRef.current.currentTime + 5, duration);
    }
  };

  const handleSkipBackward = () => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.max(audioRef.current.currentTime - 5, 0);
    }
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs < 10 ? '0' : ''}${secs}`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-yellow-100 via-pink-100 to-purple-100 relative overflow-hidden">
      <audio 
        ref={audioRef}
        onTimeUpdate={() => {
          if (audioRef.current && duration > 0) {
            setProgress(audioRef.current.currentTime / duration);
          }
        }}
        onEnded={() => {
          setIsPlaying(false);
          setProgress(1);
        }}
        onLoadedMetadata={() => {
          if (audioRef.current) {
            setDuration(audioRef.current.duration);
          }
        }}
      />
      
      <div className="absolute top-10 right-10 animate-float">
        <Sparkles className="w-8 h-8 text-purple-400 opacity-50" />
      </div>
      <div className="absolute bottom-20 right-1/4 animate-float-delayed">
        <Sparkles className="w-6 h-6 text-pink-400 opacity-60" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-6 py-8">
        <header className="flex items-center justify-between gap-4 mb-8 flex-wrap">
          <div className="flex items-center gap-3">
            <Wand2 className="w-8 h-8 md:w-10 md:h-10 text-purple-500" />
            <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold bg-gradient-to-r from-purple-600 to-pink-500 bg-clip-text text-transparent whitespace-nowrap">
              MagicNarrate
            </h1>
          </div>
          <button
            onClick={onBackToHome}
            className="flex items-center gap-2 px-4 sm:px-6 py-2 sm:py-3 bg-white/80 backdrop-blur-sm text-gray-700 rounded-full font-semibold hover:bg-white transition-all shadow-lg hover:scale-105 text-sm sm:text-base"
          >
            <Home className="w-4 h-4 sm:w-5 sm:h-5" />
            <span className="hidden sm:inline">Home</span>
          </button>
        </header>

        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          <div className="space-y-6">
            <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-xl p-8">
              <h2 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                <Sparkles className="w-6 h-6 text-yellow-500" />
                Create Your Story
              </h2>

              <div className="flex gap-4 mb-6">
                <button
                  onClick={() => setInputMode('text')}
                  className={`flex-1 py-4 px-6 rounded-2xl font-semibold transition-all flex items-center justify-center gap-2 ${
                    inputMode === 'text'
                      ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg scale-105'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  <Type className="w-5 h-5" />
                  Text Input
                </button>
                <button
                  onClick={() => setInputMode('image')}
                  className={`flex-1 py-4 px-6 rounded-2xl font-semibold transition-all flex items-center justify-center gap-2 ${
                    inputMode === 'image'
                      ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg scale-105'
                      : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                  }`}
                >
                  <ImagePlus className="w-5 h-5" />
                  Image Upload
                </button>
              </div>

              {inputMode === 'text' ? (
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Describe your story idea
                  </label>
                  <textarea
                    value={textInput}
                    onChange={(e) => setTextInput(e.target.value)}
                    placeholder="A brave little dragon who loves to bake cookies..."
                    className="w-full h-40 p-4 rounded-2xl border-2 border-purple-200 focus:border-purple-400 focus:outline-none resize-none text-lg"
                  />
                </div>
              ) : (
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Upload an image
                  </label>
                  <div className="relative">
                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleImageUpload}
                      className="hidden"
                      id="image-upload"
                      ref={fileInputRef}
                    />
                    <label
                      htmlFor="image-upload"
                      className="flex flex-col items-center justify-center h-40 border-2 border-dashed border-purple-300 rounded-2xl cursor-pointer hover:border-purple-400 transition-all hover:bg-purple-50"
                    >
                      {imagePreview ? (
                        <img
                          src={imagePreview}
                          alt="Preview"
                          className="h-full w-full object-cover rounded-2xl"
                        />
                      ) : (
                        <>
                          <ImagePlus className="w-12 h-12 text-purple-400 mb-2" />
                          <span className="text-gray-600 font-medium">
                            Click to upload
                          </span>
                        </>
                      )}
                    </label>
                  </div>
                </div>
              )}
            </div>

            <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-xl p-8">
              <h3 className="text-xl font-bold text-gray-800 mb-6">Story Settings</h3>

              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Genre
                  </label>
                  <select
                    value={genre}
                    onChange={(e) => setGenre(e.target.value)}
                    className="w-full p-4 rounded-2xl border-2 border-purple-200 focus:border-purple-400 focus:outline-none text-lg bg-white"
                  >
                    <option value="adventure">🗺️ Adventure</option>
                    <option value="fantasy">✨ Fantasy</option>
                    <option value="bedtime">🌙 Bedtime</option>
                    <option value="friendship">💕 Friendship</option>
                    <option value="learning">🔬 Learning</option>
                    <option value="confidence">💪 Confidence</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Emotion Tone
                  </label>
                  <select
                    value={emotionTone}
                    onChange={(e) => setEmotionTone(e.target.value)}
                    className="w-full p-4 rounded-2xl border-2 border-purple-200 focus:border-purple-400 focus:outline-none text-lg bg-white"
                  >
                    <option value="joyful">😊 Joyful</option>
                    <option value="funny">😄 Funny</option>
                    <option value="mysterious">🌙 Mysterious</option>
                    <option value="calm">😌 Calm</option>
                    <option value="sad">😢 Sad</option>
                    <option value="confused">😕 Confused</option>
                    <option value="dramatic">🎭 Dramatic</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    Speaker
                  </label>
                  <select
                    value={selectedSpeaker}
                    onChange={(e) => setSelectedSpeaker(e.target.value)}
                    className="w-full p-4 rounded-2xl border-2 border-purple-200 focus:border-purple-400 focus:outline-none text-lg bg-white"
                  >
                    {speakers.map((speaker) => (
                      <option key={speaker} value={speaker}>
                        {speaker}
                      </option>
                    ))}
                  </select>
                </div>

              </div>
            </div>

            <button
              onClick={handleGenerateStory}
              disabled={isGenerating || (!textInput && !imagePreview)}
              className="w-full py-5 px-8 bg-gradient-to-r from-purple-500 via-pink-500 to-yellow-400 text-white text-xl font-bold rounded-full shadow-2xl hover:shadow-purple-300 transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:scale-105 flex items-center justify-center gap-3 group"
            >
              {isGenerating ? (
                <>
                  <Wand2 className="w-6 h-6 animate-spin" />
                  Generating Magic...
                </>
              ) : (
                <>
                  <Wand2 className="w-6 h-6 group-hover:rotate-12 transition-transform" />
                  Generate Story
                  <Sparkles className="w-6 h-6" />
                </>
              )}
            </button>
          </div>

          <div className="space-y-6">
            <Suspense fallback={<div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-xl p-8 h-96 flex items-center justify-center text-gray-400">Loading...</div>}>
              <StoryResultSection
                generatedStory={generatedStory}
                isPlaying={isPlaying}
                isGeneratingAudio={isGeneratingAudio}
                progress={progress}
                duration={duration}
                selectedSpeaker={selectedSpeaker}
                onPlayPause={handlePlayPause}
                onSkipBackward={handleSkipBackward}
                onSkipForward={handleSkipForward}
                onStop={handleStop}
                onDownload={handleDownloadAudio}
                onSeek={handleSeek}
                formatTime={formatTime}
              />
            </Suspense>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SystemPage;
