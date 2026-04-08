import { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import './PredictionGauge.css';

export default function PredictionGauge({ prediction, confidence }) {
  const needleRef = useRef(null);
  const fillRef = useRef(null);

  const pred = prediction || 'Unknown';
  const conf = confidence !== undefined
    ? (confidence <= 1 ? confidence * 100 : confidence)
    : 0;

  // Correct angle mapping: 0% → -90deg, 100% → +90deg
  const angle = (conf / 100) * 180 - 90;
  const clampedAngle = Math.max(-90, Math.min(90, angle));

  // Arc fill: total half-circle arc length for r=80 ≈ 251.2
  const arcLength = 251.2;
  const dashOffset = arcLength * (1 - conf / 100);

  const getColor = () => {
    if (pred === 'High') return 'var(--red)';
    if (pred === 'Medium') return 'var(--amber)';
    return 'var(--blue)';
  };

  return (
    <div className="gauge-container">
      <div className="gauge">
        <svg viewBox="0 0 200 135" width="240">
          {/* Background arc */}
          <path
            className="gauge-arc gauge-bg"
            d="M 20 100 A 80 80 0 0 1 180 100"
          />
          {/* Dynamic fill arc */}
          <motion.path
            className="gauge-arc"
            d="M 20 100 A 80 80 0 0 1 180 100"
            style={{
              fill: 'none',
              strokeWidth: 14,
              strokeLinecap: 'round',
              stroke: getColor(),
            }}
            initial={{ strokeDasharray: arcLength, strokeDashoffset: arcLength }}
            animate={{ strokeDashoffset: dashOffset }}
            transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
          />
          {/* Needle: small triangle pointing outward, rotates around (100,100) */}
          <motion.polygon
            className="gauge-needle-tick"
            points="100,12 96,24 104,24"
            initial={{ rotate: -90 }}
            animate={{ rotate: clampedAngle }}
            transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
            style={{ transformOrigin: '100px 100px' }}
          />
          {/* Confidence text */}
          <text className="gauge-pct" x="100" y="82">
            {pred !== 'Unknown' ? `${conf.toFixed(1)}%` : ''}
          </text>
          {/* Label */}
          <text
            className="gauge-label"
            x="100" y="98"
            style={{ fill: pred === 'Unknown' ? 'var(--text-muted)' : getColor() }}
          >
            {pred === 'Unknown' ? 'AWAITING' : pred.toUpperCase()}
          </text>
          {/* Scale labels (embedded in SVG for pixel-perfect alignment) */}
          <text className="gauge-scale" x="20"  y="118" style={{ fill: 'var(--blue)' }}>Low</text>
          <text className="gauge-scale" x="100" y="125" style={{ fill: 'var(--amber)' }}>Medium</text>
          <text className="gauge-scale" x="180" y="118" style={{ fill: 'var(--red)' }}>High</text>
        </svg>
      </div>
    </div>
  );
}
