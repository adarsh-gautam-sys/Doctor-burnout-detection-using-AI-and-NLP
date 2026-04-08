import { useState, useEffect, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import UploadPanel from '../components/UploadPanel';
import PredictionGauge from '../components/PredictionGauge';
import ProbabilityBars from '../components/ProbabilityBars';
import XAIPanel from '../components/XAIPanel';
import Footer from '../components/Footer';
import { uploadAndPredict, getDatasetData } from '../services/api';
import GridScan from '../components/GridScan';
import './Home.css';

/* ═══════════════════════════════════════════════════════
   SCROLL REVEAL WRAPPER
   ═══════════════════════════════════════════════════════ */
function Reveal({ children, delay = 0, className = '' }) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 40 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-50px' }}
      transition={{ duration: 0.8, delay, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════
   ANIMATED COUNTER
   ═══════════════════════════════════════════════════════ */
function AnimatedCounter({ target, suffix = '' }) {
  const [count, setCount] = useState(0);
  const ref = useRef(null);
  const animated = useRef(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !animated.current) {
          animated.current = true;
          const duration = 1500;
          const start = performance.now();
          const tick = (now) => {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            setCount(Math.floor(target * (1 - Math.pow(1 - progress, 3))));
            if (progress < 1) requestAnimationFrame(tick);
          };
          requestAnimationFrame(tick);
        }
      },
      { threshold: 0.5 }
    );
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [target]);

  return <span ref={ref}>{count.toLocaleString()}{suffix}</span>;
}

/* ═══════════════════════════════════════════════════════
   HOME PAGE
   ═══════════════════════════════════════════════════════ */
export default function Home() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [totalImages, setTotalImages] = useState(0);

  useEffect(() => {
    getDatasetData()
      .then(d => setTotalImages(d?.summary?.total_images || 0))
      .catch(() => { });
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!selectedFile) return;
    setIsAnalyzing(true);
    setError(null);
    try {
      const result = await uploadAndPredict(selectedFile);
      setPrediction(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsAnalyzing(false);
    }
  }, [selectedFile]);

  const handleReset = () => {
    setSelectedFile(null);
    setPrediction(null);
    setError(null);
  };

  const handleExportTxt = () => {
    if (!prediction) return;
    const pred = prediction.burnout_risk || prediction.prediction || 'Unknown';
    const conf = prediction.confidence !== undefined
      ? (prediction.confidence <= 1 ? prediction.confidence * 100 : prediction.confidence)
      : 0;

    let text = `======================================\n`;
    text += ` BurnoutAI — Prescription Analysis\n`;
    text += ` Date: ${new Date().toLocaleString()}\n`;
    text += `======================================\n\n`;
    text += `Burnout Risk Level: ${pred.toUpperCase()}\n`;
    text += `Confidence Level:   ${conf.toFixed(1)}%\n\n`;

    if (prediction.explanation?.top_features) {
      text += `[ AI EXPLANATION SUMMARY ]\n`;
      text += `${prediction.explanation.summary || 'Analysis complete.'}\n\n`;
      text += `[ FEATURE IMPACT ANALYSIS ]\n`;
      prediction.explanation.top_features.forEach(f => {
        text += `- ${f.display_name}: ${(f.importance * 100).toFixed(1)}%\n`;
      });
      text += `\n`;
      if (prediction.explanation.findings?.length > 0) {
        text += `[ KEY FINDINGS ]\n`;
        prediction.explanation.findings.forEach(f => { text += `- ${f}\n`; });
      }
    }

    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `burnout_analysis_${new Date().toISOString().slice(0, 10)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const pred = prediction?.burnout_risk || prediction?.prediction || 'Unknown';
  const conf = prediction?.confidence !== undefined
    ? (prediction.confidence <= 1 ? prediction.confidence * 100 : prediction.confidence)
    : 0;
  const probs = prediction?.probabilities || {};

  return (
    <>
      {/* ═══ HERO ═══ */}
      <section id="hero" className="hero">
        <GridScan
          sensitivity={0.55}
          lineThickness={1}
          linesColor="#2a2a2a"
          gridScale={0.1}
          scanColor="#f59e0b"
          scanOpacity={0.4}
          enablePost
          bloomIntensity={0.6}
          chromaticAberration={0.002}
          noiseIntensity={0.01}
        />

        <div className="hero__risk-readout">RISK LEVEL: MONITORING...</div>

        <div className="container">
          <div className="hero__content">
            <Reveal>
              <h1 className="hero__headline">
                Burnout Begins Before the <em>Breaking Point.</em>
              </h1>
            </Reveal>
            <Reveal delay={0.1}>
              <p className="hero__sub">
                An AI and NLP pipeline that reads a doctor's handwriting to detect burnout risk —
                automatically, before errors happen.
              </p>
            </Reveal>
            <Reveal delay={0.2}>
              <div className="hero__buttons">
                <a href="#demo" className="btn-primary">Try the Demo</a>
              </div>
            </Reveal>
          </div>
        </div>
      </section>

      {/* ═══ THE PROBLEM ═══ */}
      <section id="problem" className="section section--dim">
        <div className="container">
          <Reveal><span className="section-label">The Problem</span></Reveal>
          <Reveal delay={0.05}><h2 className="section-title">A crisis hiding in plain sight.</h2></Reveal>
          <div className="stat-grid">
            <Reveal delay={0.1}>
              <div className="stat-card glass-card">
                <div className="stat-number">40–60%</div>
                <p className="stat-label">Physicians worldwide report symptoms of burnout — emotional exhaustion, depersonalization, or reduced personal accomplishment.</p>
              </div>
            </Reveal>
            <Reveal delay={0.2}>
              <div className="stat-card glass-card">
                <div className="stat-number" style={{ fontSize: '1.8rem', lineHeight: 1.2, marginBottom: '1rem' }}>Delayed Detection</div>
                <p className="stat-label">Existing burnout detection relies on surveys and self-reporting, which are delayed and subjective.</p>
              </div>
            </Reveal>
            <Reveal delay={0.3}>
              <div className="stat-card glass-card">
                <div className="stat-number">
                  {totalImages > 0 ? <AnimatedCounter target={totalImages} /> : '—'}
                </div>
                <p className="stat-label">Prescription images analyzed across real and synthetic document sources.</p>
              </div>
            </Reveal>
          </div>
        </div>
      </section>

      {/* ═══ THE INSIGHT ═══ */}
      <section id="insight" className="section">
        <div className="container" style={{ textAlign: 'center' }}>
          <Reveal>
            <p className="insight-quote">
              "Burnt-out doctors write differently. Shorter sentences. Shakier lines.
              More abbreviations. Less detail. <em>This is a signal — and it's been invisible.
                Until now.</em>"
            </p>
          </Reveal>
          <div className="insight-cols">
            {[
              { title: 'Handwriting Degradation', text: 'Motor control degrades under chronic stress. Stroke pressure drops, letter consistency erodes, and spatial regularity declines — measurable changes that an OCR pipeline can quantify.' },
              { title: 'Linguistic Shortcuts', text: "Burnt-out physicians use more abbreviations, shorter sentences, and fewer diagnostic qualifiers. NLP analysis detects these patterns as a ratio change over time." },
              { title: 'Negative Sentiment Drift', text: 'Clinical notes shift in tone — less patient-centered language, more terseness, fewer empathetic markers. Sentiment analysis captures this gradual drift.' },
            ].map((col, i) => (
              <Reveal key={col.title} delay={i * 0.1}>
                <div className="insight-col glass-card">
                  <h3>{col.title}</h3>
                  <p>{col.text}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* ═══ DEMO ═══ */}
      <section id="demo" className="section">
        <div className="container">
          <Reveal><span className="section-label">Live Demo</span></Reveal>
          <Reveal delay={0.05}><h2 className="section-title">See it in action.</h2></Reveal>

          <Reveal delay={0.1}>
            <div className="demo-card glass-card">
              <h3 style={{ fontFamily: 'var(--font-display)', fontSize: '1.6rem', marginBottom: '0.5rem' }}>
                Analyze a Prescription
              </h3>
              <p className="demo-subtitle">
                Upload a handwritten prescription image to get an instant burnout risk assessment.
              </p>

              <UploadPanel
                onFileSelect={setSelectedFile}
                isAnalyzing={isAnalyzing}
              />

              <PredictionGauge prediction={pred} confidence={conf} />

              <button
                className={selectedFile && !isAnalyzing ? 'btn-primary' : 'btn-primary'}
                disabled={!selectedFile || isAnalyzing}
                onClick={handleAnalyze}
                style={{ width: '100%', marginTop: '1.5rem' }}
              >
                {isAnalyzing ? 'Analyzing...' : 'Analyze'}
              </button>

              <ProbabilityBars probabilities={probs} />

              {/* Error */}
              {error && (
                <motion.div
                  className="demo-error"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                >
                  <strong>⚠ Could not reach model at localhost:8000.</strong><br />
                  Make sure your FastAPI server is running.
                </motion.div>
              )}

              {/* Reset & Export */}
              {prediction && (
                <motion.div
                  className="demo-actions"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.3 }}
                >
                  <button className="btn-link" onClick={handleReset}>
                    Analyze another prescription →
                  </button>
                  <button className="btn-outline" onClick={handleExportTxt} style={{ padding: '0.4rem 1rem', fontSize: '0.75rem' }}>
                    ⬇ Export Report (TXT)
                  </button>
                </motion.div>
              )}
            </div>
          </Reveal>

          {/* XAI Panel */}
          {prediction?.explanation && (
            <XAIPanel explanation={prediction.explanation} />
          )}
        </div>
      </section>

      {/* ═══ TECH STACK ═══ */}
      <section id="stack" className="section section--dim">
        <div className="container">
          <Reveal><span className="section-label">Technology</span></Reveal>
          <Reveal delay={0.05}><h2 className="section-title">Tech Stack</h2></Reveal>

          <div className="stack-grid">
            {[
              { category: 'Deep Learning', items: ['PyTorch 2.10.0', 'torchvision 0.25.0', 'CUDA 12.8', 'Tesla T4 GPU'] },
              { category: 'OCR Engines', items: ['TrOCR (microsoft)', 'EasyOCR 1.7.2', 'Tesseract 4.1.1', 'pytesseract'] },
              { category: 'NLP & Layout', items: ['HuggingFace Transformers 5.0.0', 'LayoutLMv3', 'MarianMT', 'jiwer', 'langdetect'] },
              { category: 'Infrastructure', items: ['FastAPI', 'Gradio', 'Celery', 'Redis', 'Docker', 'PostgreSQL', 'Prometheus'] },
            ].map((group, gi) => (
              <Reveal key={group.category} delay={gi * 0.1}>
                <div className="stack-category">
                  <h3>{group.category}</h3>
                  <div className="stack-badges">
                    {group.items.map(item => (
                      <span className="stack-badge" key={item}>{item}</span>
                    ))}
                  </div>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* ═══ ROADMAP ═══ */}
      <section id="next" className="section">
        <div className="container">
          <Reveal><span className="section-label">Roadmap</span></Reveal>
          <Reveal delay={0.05}><h2 className="section-title">What's Next</h2></Reveal>

          <div className="roadmap-grid">
            {[
              { title: 'Scale the Dataset', text: '200 training samples are insufficient for a 333.9M parameter model. Target: 10,000+ annotated medical document images.' },
              { title: 'Hospital Pilot', text: 'The FastAPI + Docker stack is production-ready. Partner with a teaching hospital to deploy the pipeline on real prescriptions.' },
              { title: 'Longitudinal Tracking', text: "Track per-doctor handwriting and linguistic patterns over weeks and months. Detect drift toward burnout indicators." },
            ].map((item, i) => (
              <Reveal key={item.title} delay={i * 0.1}>
                <div className="roadmap-card glass-card">
                  <h3>{item.title}</h3>
                  <p>{item.text}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <Footer />
    </>
  );
}
