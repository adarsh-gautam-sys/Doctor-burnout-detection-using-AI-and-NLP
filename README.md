# 🏥 BurnoutAI (CliniCare v2)

An advanced multimodal artificial intelligence system designed to detect clinical burnout in healthcare professionals through the analysis of handwritten prescriptions and medical documents.

## 🌟 The V5 Multimodal Pipeline
Currently running the **V5 Model (87.6% Accurate)**, completely transitioning away from legacy heuristics. The pipeline achieves state-of-the-art accuracy by combining three distinct feature extraction modalities:
1. **Vision (PyTorch EfficientNet-B3):** Extracts 1,536 deep latent spatial features from the raw image.
2. **NLP (EasyOCR):** Extracts 20 linguistic dimensions (medical shorthand frequency, word fragmentation, syntax deterioration).
3. **Handcrafted Heuristics:** Maps 36 structural traits (ink density drops, baseline shifts).

Features are organically fused and dimension-reduced via **ANOVA F-value SelectKBest (128 features)** to train an aggressive SVM classifier alongside an interactive XAI (Explainable AI) engine.

---

## 📂 Repository Structure

The workspace is modularized for production deployment:

- `/frontend` → The robust React 19 + Vite UI. Powered by premium animations and components from Shadcn UI & ReactBits.
- `/backend` → FastAPI Inference Engine holding `predict.py` and actively loading the `V5` PyTorch artifact dependencies from `/data/models`.
- `/ml_pipeline` → Isolated ML training pipeline housing `train_model_v5.py` orchestrating MixUp augmentations and Consensus Tri-Training pseudo-labeling. 
- `/notebooks` → Academic research literature, historical EDA, and Jupyter notebooks.
- `/legacy_ui` → The old vanilla HTML/JS prototypes (preserved for reference).

---

## 🛠️ Quick Start & Installation

Ensure you have **Python 3.10+** and **Node.js** installed on your machine.

### 1. Start the Backend API
First, install the Python dependencies and launch the FastAPI server.
```bash
# Create and activate a Virtual Environment
python -m venv venv
.\venv\Scripts\activate   # On Windows
source venv/bin/activate  # On macOS/Linux

# Install dependencies (Downloads PyTorch, OpenCV, EasyOCR, etc.)
pip install -r requirements.txt

# Start the Backend Inference Engine
cd backend
python predict.py
```
*(The backend runs at `http://localhost:8000` and automatically loads the GPU CUDA pipeline if available).*

### 2. Start the Frontend Application
In a separate terminal, start the React interface:
```bash
cd frontend

# Install Node modules
npm install

# Start the Vite development server
npm run dev
```
*(The UI runs at `http://localhost:5173`)*

---

## 🔬 Explainable AI (XAI)
BurnoutAI isn't a black box. The backend generates interactive feature-weight explanations for every prediction, rendering dynamic insights to the frontend (e.g. *"High clinical shorthand density correlated heavily with extreme fatigue"*).

## 📄 License
This system is part of an academic and private research initiative. All rights reserved by the respective authors.
