import { useEffect, useRef } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Dashboard from './pages/Dashboard';

import BlobCursor from './components/BlobCursor';

/* ═══════════════════════════════════════════════════════
   SCROLL TO TOP ON ROUTE CHANGE
   ═══════════════════════════════════════════════════════ */
function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
}

/* ═══════════════════════════════════════════════════════
   APP
   ═══════════════════════════════════════════════════════ */
export default function App() {
  return (
    <Router>
      <BlobCursor fillColor="#f59e0b" zIndex={9999} />
      <ScrollToTop />
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/dashboard" element={<Dashboard />} />
      </Routes>
    </Router>
  );
}
