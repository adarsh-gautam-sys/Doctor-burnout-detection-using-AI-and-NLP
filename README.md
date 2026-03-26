<div align="center">
  <img src="./data/docs/header.png" alt="BurnoutAI Header" width="100%">
  
  # **BurnoutAI (CliniCare v2)**
  
  **An AI & NLP Pipeline that detects physician burnout from handwritten prescriptions — automatically, before errors happen.**

  [![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
  [![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com)
  [![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
  [![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-blue.svg)](https://xgboost.readthedocs.io/)
</div>

<br>

## 💡 The Problem: A Crisis Hiding in Plain Sight
Existing burnout detection relies on surveys and self-reporting, which are delayed and subjective. By the time burnout is identified, it often already impacts clinical performance through medical errors, terseness, or reduced empathy.

**The Insight:** Burnt-out doctors write differently. Motor control degrades under chronic stress. Shorter sentences. Shakier lines. More abbreviations. Less detail. *This is a signal — and our pipeline quantifies it.*

---

## 🚀 Features

* **End-to-End Pipeline:** Ingests document images, pre-processes them, and runs hybrid feature extraction (Visual + NLP).
* **490-Dimensional Feature Extraction:** Analyzes handwriting degradation (ink density, stroke fragmentation, edge irregularity, writing pressure) alongside linguistic shortcuts via OCR.
* **Explainable AI (XAI):** Doesn't just give a risk score. Explains *why* using model-extracted feature importances mapped to human-readable clinical findings.
* **Standalone Hospital Dashboard:** A beautifully designed frontend for administrators to monitor hospital-wide risk distributions, sort doctors by risk, and identify critical interventions.
* **Instant Export:** One-click CSV/TXT generation of physician risk reports equipped with AI-generated mitigation advice.

---

## 🧠 Architecture & Tech Stack

The architecture splits into a lightweight modern Vanilla JS frontend and a robust Python backend inference engine.

### Core Stack
* **Deep Learning / NLP:** PyTorch, HuggingFace Transformers, XGBoost, Scikit-Learn.
* **OCR Engines:** EasyOCR (best on degraded text), Microsoft TrOCR, Tesseract.
* **Backend:** FastAPI (serving the `/predict`, `/ocr`, and `/generate_report` endpoints).
* **Frontend:** Vanilla HTML5, CSS3 (Custom Properties, Glassmorphism), JavaScript (ES6+).

---

## 🛠️ Usage & Setup

### 1. Clone the repository
```bash
git clone https://github.com/adarsh-gautam-sys/Doctor-burnout-detection-using-AI-and-NLP.git
cd Doctor-burnout-detection-using-AI-and-NLP
```

### 2. Setup the Python Environment
Ensure you have Python 3.11 installed. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

*(Note: `torch` installation may require specific CUDA versions depending on your hardware. Visit [PyTorch](https://pytorch.org/get-started/locally/) for specific commands).*

### 3. Run the Backend API
Start the FastAPI inference server containing the XGBoost model:
```bash
python predict.py
```
*The server will start on `http://localhost:8000`.*

### 4. Run the Client Dashboard
In a new terminal, serve the frontend:
```bash
python -m http.server 8080
```
Open your browser and navigate to `http://localhost:8080/index.html` to access the Live Demo and Dashboard.

---

## 📊 Directory Structure

```text
├── data/                    # Contains required JSON feeds, dataset samples, and pickled models
│   ├── dashboard_data.json  # Fetched by the standalone dashboard
│   ├── dataset.json
│   ├── models/              # Pretrained XGBoost/SVM model weights (.pkl)
│   └── labeled/             # Image dataset (Real and Synthetic)
├── index.html               # Main landing page & Live Demo with XAI Panel
├── dashboard.html           # Hospital-wide statistics and reporting interface
├── predict.py               # FastAPI backend with XAI feature extraction pipeline
├── train_model.py           # Logic for training the XGBoost model on the 490 feature dimensions
├── organize_dataset.py      # Utility for restructuring the raw labels/images
├── Handwritten_data.ipynb   # Exploratory Data Analysis (EDA) on the prescription samples
├── burnout_v3_semisupervised.ipynb # Complete semi-supervised model training notebook
└── README.md
```

<br>

<div align="center">
  <sub>Built for clinical excellence. Powered by AI and NLP.</sub>
</div>
