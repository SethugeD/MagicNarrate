import { useState, Suspense, lazy } from 'react';

const LandingPage = lazy(() => import('./components/LandingPage'));
const SystemPage = lazy(() => import('./components/SystemPage'));

function App() {
  const [currentPage, setCurrentPage] = useState<'landing' | 'system'>('landing');

  return (
    <main className="min-h-screen">
      <Suspense fallback={<div className="flex items-center justify-center h-screen">Loading...</div>}>
        {currentPage === 'landing' ? (
          <LandingPage onGetStarted={() => setCurrentPage('system')} />
        ) : (
          <SystemPage onBackToHome={() => setCurrentPage('landing')} />
        )}
      </Suspense>
    </main>
  );
}

export default App;
