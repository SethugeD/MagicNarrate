import { Sparkles, Upload, Wand2, Headphones, BookOpen } from 'lucide-react';

interface LandingPageProps {
  onGetStarted: () => void;
}

function LandingPage({ onGetStarted }: LandingPageProps) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-100 via-purple-100 to-blue-100 relative overflow-hidden">
      <div className="absolute top-10 left-10 animate-float">
        <Sparkles className="w-8 h-8 text-yellow-400 opacity-60" />
      </div>
      <div className="absolute top-40 right-20 animate-float-delayed">
        <Sparkles className="w-6 h-6 text-pink-400 opacity-60" />
      </div>
      <div className="absolute bottom-32 left-20 animate-float">
        <Sparkles className="w-10 h-10 text-purple-400 opacity-40" />
      </div>
      <div className="absolute top-1/2 right-10 animate-float-delayed">
        <Sparkles className="w-7 h-7 text-blue-400 opacity-50" />
      </div>

      <div className="relative z-10 max-w-6xl mx-auto px-6 py-16">
        <header className="text-center mb-16 animate-fade-in">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Wand2 className="w-12 h-12 text-purple-500" />
            <h1 className="text-6xl font-bold bg-gradient-to-r from-purple-600 via-pink-500 to-blue-500 bg-clip-text text-transparent pb-1 leading-tight">
              MagicNarrate
            </h1>
          </div>
          <p className="text-2xl text-gray-700 font-medium">
            Bring your stories to life with AI-powered imagination
          </p>
        </header>

        <section className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-12 mb-12 animate-slide-up">
          <div className="flex items-center gap-3 mb-6">
            <BookOpen className="w-8 h-8 text-purple-500" />
            <h2 className="text-3xl font-bold text-gray-800">About</h2>
          </div>
          <p className="text-xl text-gray-700 leading-relaxed">
            MagicNarrate is an AI-powered storytelling companion that transforms your images and text into beautifully narrated stories.
            Whether you upload a picture or describe a scene, our magical AI creates enchanting tales that you can read and listen to!
          </p>
        </section>

        <section className="bg-white/80 backdrop-blur-sm rounded-3xl shadow-xl p-12 mb-16 animate-slide-up-delayed">
          <h2 className="text-3xl font-bold text-gray-800 text-center mb-12">How It Works</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center group">
              <div className="bg-gradient-to-br from-purple-200 to-purple-300 w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300 shadow-lg">
                <Upload className="w-12 h-12 text-purple-700" />
              </div>
              <h3 className="text-2xl font-bold text-gray-800 mb-3">1. Input</h3>
              <p className="text-gray-600 text-lg">
                Upload an image or type your story idea
              </p>
            </div>

            <div className="text-center group">
              <div className="bg-gradient-to-br from-pink-200 to-pink-300 w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300 shadow-lg">
                <Wand2 className="w-12 h-12 text-pink-700" />
              </div>
              <h3 className="text-2xl font-bold text-gray-800 mb-3">2. Generate</h3>
              <p className="text-gray-600 text-lg">
                AI creates a magical story just for you
              </p>
            </div>

            <div className="text-center group">
              <div className="bg-gradient-to-br from-blue-200 to-blue-300 w-24 h-24 rounded-full flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300 shadow-lg">
                <Headphones className="w-12 h-12 text-blue-700" />
              </div>
              <h3 className="text-2xl font-bold text-gray-800 mb-3">3. Listen</h3>
              <p className="text-gray-600 text-lg">
                Read along or listen to the narrated tale
              </p>
            </div>
          </div>
        </section>

        <div className="text-center animate-bounce-slow">
          <button
            onClick={onGetStarted}
            className="group relative px-12 py-5 text-2xl font-bold text-white bg-gradient-to-r from-purple-500 via-pink-500 to-blue-500 rounded-full shadow-2xl hover:shadow-purple-300 transition-all duration-300 hover:scale-105"
          >
            <span className="relative z-10 flex items-center gap-3">
              Get Started
              <Sparkles className="w-6 h-6 group-hover:rotate-12 transition-transform" />
            </span>
            <div className="absolute inset-0 bg-gradient-to-r from-purple-600 via-pink-600 to-blue-600 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-300 blur-xl" />
          </button>
        </div>
      </div>

      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-white/20 to-transparent" />
    </div>
  );
}

export default LandingPage;
