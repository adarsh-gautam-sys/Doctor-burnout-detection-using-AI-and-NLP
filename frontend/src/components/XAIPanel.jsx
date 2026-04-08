import { motion } from 'framer-motion';
import './XAIPanel.css';

export default function XAIPanel({ explanation }) {
  if (!explanation) return null;

  const { summary, model_type, feature_count, interpretable_features_analyzed, top_features, findings } = explanation;

  return (
    <motion.div
      className="xai-panel"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
    >
      {/* Summary Card */}
      <div className="xai-card glass-card">
        <div className="xai-card-header">AI Explanation Summary</div>
        <div className="xai-summary-box">{summary || 'Analysis complete.'}</div>
        <div className="xai-meta">
          <span>Model: {model_type || 'Unknown'}</span>
          <span>Features analyzed: {feature_count || 490}</span>
          <span>Interpretable: {interpretable_features_analyzed || 106}</span>
        </div>
      </div>

      {/* Feature Impact */}
      {top_features && top_features.length > 0 && (
        <div className="xai-card glass-card">
          <div className="xai-card-header">Feature Impact Analysis</div>
          {top_features.map((feat, idx) => {
            const barWidth = Math.max(feat.importance * 100, 5);
            return (
              <div className="xai-feature-row" key={feat.feature_name}>
                <span className="xai-feature-name">{feat.display_name}</span>
                <div className="xai-bar-bg">
                  <motion.div
                    className="xai-bar-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${barWidth}%` }}
                    transition={{ duration: 0.8, delay: idx * 0.12, ease: [0.16, 1, 0.3, 1] }}
                  />
                </div>
                <span className="xai-feature-value">{(feat.importance * 100).toFixed(0)}%</span>
              </div>
            );
          })}
        </div>
      )}

      {/* Key Findings */}
      {findings && findings.length > 0 && (
        <div className="xai-card glass-card">
          <div className="xai-card-header">Key Findings</div>
          {findings.map((finding, idx) => (
            <motion.div
              className="xai-finding"
              key={idx}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.4, delay: 0.3 + idx * 0.1 }}
            >
              <div className="xai-finding-dot" />
              <span className="xai-finding-text">{finding}</span>
            </motion.div>
          ))}
        </div>
      )}
    </motion.div>
  );
}
