import { motion } from 'framer-motion';
import './ProbabilityBars.css';

export default function ProbabilityBars({ probabilities }) {
  const probs = probabilities || {};
  const lowP = (probs.Low || 0) * 100;
  const medP = (probs.Medium || 0) * 100;
  const highP = (probs.High || 0) * 100;

  const bars = [
    { label: 'Low Risk Probability', value: lowP, color: 'var(--green)' },
    { label: 'Medium Risk Probability', value: medP, color: 'var(--amber)' },
    { label: 'High Risk Probability', value: highP, color: 'var(--red)' },
  ];

  return (
    <div className="probability-bars">
      {bars.map((bar, i) => (
        <div className="prob-row" key={bar.label}>
          <span className="prob-label">{bar.label}</span>
          <div className="prob-bar-bg">
            <motion.div
              className="prob-bar-fill"
              style={{ background: bar.color }}
              initial={{ width: 0 }}
              animate={{ width: `${bar.value}%` }}
              transition={{ duration: 0.8, delay: i * 0.1, ease: [0.16, 1, 0.3, 1] }}
            />
          </div>
          <span className="prob-value">{bar.value.toFixed(1)}%</span>
        </div>
      ))}
    </div>
  );
}
