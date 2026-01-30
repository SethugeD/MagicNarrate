import { Sparkles } from 'lucide-react';

function AICharacter() {
  return (
    <div className="fixed bottom-8 right-8 z-50 animate-float group cursor-pointer">
      <div className="relative">
        <div className="absolute -inset-2 bg-gradient-to-r from-purple-400 to-pink-400 rounded-full opacity-60 group-hover:opacity-100 blur-lg transition-opacity" />
        <div className="relative bg-gradient-to-br from-purple-500 to-pink-500 w-20 h-20 rounded-full flex items-center justify-center shadow-2xl">
          <Sparkles className="w-10 h-10 text-white group-hover:rotate-12 transition-transform" />
        </div>

        <div className="absolute -top-16 right-0 bg-white rounded-2xl px-4 py-2 shadow-xl opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
          <p className="text-sm font-semibold text-purple-600">
            Need help? I'm here! ✨
          </p>
          <div className="absolute bottom-0 right-8 transform translate-y-1/2 rotate-45 w-3 h-3 bg-white" />
        </div>
      </div>
    </div>
  );
}

export default AICharacter;
