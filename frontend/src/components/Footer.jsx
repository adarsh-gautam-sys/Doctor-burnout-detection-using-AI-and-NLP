import './Footer.css';

export default function Footer() {
  return (
    <footer className="footer">
      <div className="container">
        <div className="footer__inner">
          <div className="footer__brand">
            Burnout<span>AI</span> — AI &amp; NLP Research Project
          </div>
          <div className="footer__meta">
            <a
              href="https://github.com/adarsh-gautam-sys/Doctor-burnout-detection-using-AI-and-NLP"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub
            </a>
            <span>Built with PyTorch · HuggingFace · EasyOCR</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
