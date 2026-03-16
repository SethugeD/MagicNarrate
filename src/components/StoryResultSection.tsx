import { useRef, useEffect } from 'react';
import { Sparkles, Volume2, Play, Pause, SkipBack, SkipForward, RotateCcw, Download, Wand2 } from 'lucide-react';

interface StoryResultSectionProps {
  generatedStory: string;
  isPlaying: boolean;
  isGeneratingAudio: boolean;
  audioStatusMessage: string;
  progress: number;
  duration: number;
  selectedSpeaker: string;
  onPlayPause: () => void;
  onSkipBackward: () => void;
  onSkipForward: () => void;
  onStop: () => void;
  hasAudio: boolean;
  onStoryChange: (story: string) => void;
  onRegenerateAudio: () => void;
  onDownload: () => void;
  onSeek: (e: React.ChangeEvent<HTMLInputElement>) => void;
  formatTime: (seconds: number) => string;
}

export default function StoryResultSection({
  generatedStory,
  isPlaying,
  isGeneratingAudio,
  audioStatusMessage,
  progress,
  duration,
  selectedSpeaker,
  onPlayPause,
  onSkipBackward,
  onSkipForward,
  onStop,
  hasAudio,
  onStoryChange,
  onRegenerateAudio,
  onDownload,
  onSeek,
  formatTime,
}: StoryResultSectionProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${el.scrollHeight}px`;
  }, [generatedStory]);

  return (
    <div className="space-y-6">
      <div className="bg-white/90 backdrop-blur-sm rounded-3xl shadow-xl p-8 min-h-[440px] lg:min-h-[560px] flex flex-col">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
            <Sparkles className="w-6 h-6 text-pink-500" />
            Your Story
          </h2>
        </div>

        {generatedStory ? (
          <div className="mb-8">
            {hasAudio && (
              <label className="block text-sm font-semibold text-gray-700 mb-2">
                Edit story text, then regenerate audio below
              </label>
            )}
            <textarea
              ref={textareaRef}
              value={generatedStory}
              onChange={(e) => onStoryChange(e.target.value)}
              readOnly={isGeneratingAudio}
              rows={1}
              className={`w-full text-lg text-gray-700 leading-relaxed p-4 rounded-2xl border-2 transition-colors resize-none overflow-hidden focus:outline-none text-justify ${
                isGeneratingAudio
                  ? 'border-purple-100 bg-gray-50 text-gray-500 cursor-not-allowed'
                  : 'border-purple-100 focus:border-purple-300 bg-white'
              }`}
            />
          </div>
        ) : (
          <div className="flex-1 flex items-center justify-center text-gray-400">
            <div className="text-center">
              <Wand2 className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p className="text-lg">Your magical story will appear here...</p>
            </div>
          </div>
        )}

        {generatedStory && (
          <div className="border-t-2 border-purple-100 pt-6">
            <div className="flex items-center gap-2 mb-6">
              <Volume2 className="w-5 h-5 text-purple-500" />
              <span className="text-sm font-semibold text-gray-700">Audio Player</span>
              <span className="text-sm text-gray-500">• Voice: {selectedSpeaker}</span>
              {isGeneratingAudio && (
                <span className="text-sm text-purple-600 font-semibold animate-pulse">
                  {audioStatusMessage || 'Generating audio...'}
                </span>
              )}
            </div>

            {isGeneratingAudio ? (
              <div className="flex items-center justify-center py-8">
                <div className="text-center">
                  <div className="inline-block">
                    <Sparkles className="w-8 h-8 text-purple-500 animate-spin" />
                  </div>
                  <p className="text-gray-600 mt-3 font-medium">{audioStatusMessage || 'Generating audio narration...'}</p>
                </div>
              </div>
            ) : hasAudio ? (
              <>
                <div className="mb-6">
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={progress}
                    onChange={onSeek}
                    className="w-full h-2 bg-gradient-to-r from-purple-200 to-pink-200 rounded-full appearance-none cursor-pointer slider"
                    style={{
                      background: `linear-gradient(to right, rgb(168, 85, 247) 0%, rgb(236, 72, 153) ${progress * 100}%, rgb(243, 232, 255) ${progress * 100}%, rgb(243, 232, 255) 100%)`,
                    }}
                  />
                  <div className="flex justify-between text-xs text-gray-600 mt-2">
                    <span>{formatTime(progress * duration)}</span>
                    <span>{formatTime(duration)}</span>
                  </div>
                </div>

                <div className="flex gap-2 items-center justify-center mb-4">
                  <button
                    onClick={onSkipBackward}
                    className="p-3 bg-gray-200 text-gray-700 rounded-full hover:bg-gray-300 transition-all shadow-lg hover:scale-110"
                  >
                    <SkipBack className="w-5 h-5" />
                  </button>
                  <button
                    onClick={onPlayPause}
                    className="flex-1 py-4 px-6 bg-gradient-to-r from-blue-400 to-blue-500 text-white rounded-full font-semibold hover:scale-105 transition-all shadow-lg flex items-center justify-center gap-2 group"
                  >
                    {isPlaying ? (
                      <>
                        <Pause className="w-5 h-5 group-hover:scale-110 transition-transform" />
                        Pause
                      </>
                    ) : (
                      <>
                        <Play className="w-5 h-5 group-hover:scale-110 transition-transform" />
                        Play
                      </>
                    )}
                  </button>
                  <button
                    onClick={onSkipForward}
                    className="p-3 bg-gray-200 text-gray-700 rounded-full hover:bg-gray-300 transition-all shadow-lg hover:scale-110"
                  >
                    <SkipForward className="w-5 h-5" />
                  </button>
                </div>

                <button
                  onClick={onStop}
                  className="w-full py-3 px-6 bg-gray-200 text-gray-700 rounded-full font-semibold hover:bg-gray-300 transition-all shadow-lg flex items-center justify-center gap-2 group"
                >
                  <RotateCcw className="w-5 h-5 group-hover:rotate-12 transition-transform" />
                  Replay
                </button>

                <button
                  onClick={onDownload}
                  className="w-full mt-3 py-3 px-6 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-full font-semibold hover:scale-105 transition-all shadow-lg flex items-center justify-center gap-2 group"
                >
                  <Download className="w-5 h-5 group-hover:-translate-y-0.5 transition-transform" />
                  Download Audio
                </button>

                <button
                  onClick={onRegenerateAudio}
                  disabled={!generatedStory.trim()}
                  className="w-full mt-3 py-3 px-6 bg-gradient-to-r from-indigo-500 to-purple-500 text-white rounded-full font-semibold hover:scale-105 transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  <RotateCcw className="w-4 h-4" />
                  Regenerate Audio From Edited Story
                </button>
              </>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}
