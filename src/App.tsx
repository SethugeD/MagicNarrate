import { useState } from 'react';
import LandingPage from './components/LandingPage';
import SystemPage from './components/SystemPage';

function App() {
  const [currentPage, setCurrentPage] = useState<'landing' | 'system'>('landing');

  return (
    <div className="min-h-screen">
      {currentPage === 'landing' ? (
        <LandingPage onGetStarted={() => setCurrentPage('system')} />
      ) : (
        <SystemPage onBackToHome={() => setCurrentPage('landing')} />
      )}
    </div>
  );
}

export default App;
