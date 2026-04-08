import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import './Navbar.css';

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 60);
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const isActive = (path) => location.pathname === path;

  return (
    <motion.nav
      className={`navbar ${scrolled ? 'navbar--scrolled' : ''}`}
      initial={{ y: -80 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
    >
      <Link to="/" className="navbar__logo">
        Burnout<span>AI</span>
      </Link>

      <button
        className="navbar__hamburger"
        onClick={() => setMobileOpen(!mobileOpen)}
        aria-label="Toggle menu"
      >
        <span />
        <span />
        <span />
      </button>

      <ul className={`navbar__links ${mobileOpen ? 'navbar__links--open' : ''}`}>
        <li>
          <Link
            to="/"
            className={isActive('/') ? 'active' : ''}
            onClick={() => setMobileOpen(false)}
          >
            Home
          </Link>
        </li>
        <li>
          <a
            href={location.pathname === '/' ? '#demo' : '/#demo'}
            onClick={() => setMobileOpen(false)}
          >
            Demo
          </a>
        </li>
        <li>
          <Link
            to="/dashboard"
            className={isActive('/dashboard') ? 'active' : ''}
            onClick={() => setMobileOpen(false)}
          >
            Dashboard
          </Link>
        </li>
      </ul>
    </motion.nav>
  );
}
