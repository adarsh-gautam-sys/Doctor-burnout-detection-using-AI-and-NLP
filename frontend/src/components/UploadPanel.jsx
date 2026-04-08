import { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './UploadPanel.css';

export default function UploadPanel({ onFileSelect, isAnalyzing }) {
  const [dragActive, setDragActive] = useState(false);
  const [fileName, setFileName] = useState(null);
  const inputRef = useRef(null);

  const handleFile = useCallback((file) => {
    setFileName(file.name);
    onFileSelect(file);
  }, [onFileSelect]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  }, [handleFile]);

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragActive(true);
  };

  const handleDragLeave = () => setDragActive(false);

  const handleInputChange = (e) => {
    if (e.target.files?.[0]) {
      handleFile(e.target.files[0]);
    }
  };

  return (
    <motion.div
      className={`upload-zone ${dragActive ? 'upload-zone--active' : ''}`}
      onClick={() => inputRef.current?.click()}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      whileHover={{ borderColor: 'rgba(245,158,11,0.6)' }}
      transition={{ duration: 0.2 }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/*,.txt"
        onChange={handleInputChange}
        hidden
      />

      <svg
        className="upload-icon"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" y1="3" x2="12" y2="15" />
      </svg>

      <AnimatePresence mode="wait">
        <motion.p
          key={fileName || 'default'}
          className="upload-text"
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.2 }}
        >
          {isAnalyzing
            ? 'Analyzing...'
            : fileName
              ? `Selected: ${fileName}`
              : 'Drop prescription image here or click to browse'}
        </motion.p>
      </AnimatePresence>
    </motion.div>
  );
}
