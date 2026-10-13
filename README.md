# BurnoutAI (CliniCare v2)

An advanced multimodal artificial intelligence system designed to detect clinical burnout in healthcare professionals through the analysis of handwritten prescriptions and medical documents.

---

## 🌐 Live Production Deployments

The system is deployed and serving traffic globally on Google Cloud Platform and Firebase:

| Service | Hosting Platform | Production URL | Status |
|:---|:---|:---|:---:|
| **Frontend Web App** | **Firebase Hosting** | [https://clinicareai.web.app](https://clinicareai.web.app) | **Online** ✅ |
| **Mirror Domain** | **Firebase Hosting** | [https://clinicareai.firebaseapp.com](https://clinicareai.firebaseapp.com) | **Online** ✅ |
| **Backend API** | **Google Cloud Run** | [https://clinicare-backend-587520969884.us-central1.run.app](https://clinicare-backend-587520969884.us-central1.run.app) | **Online** ✅ |
| **API Health Check** | **Google Cloud Run** | [https://clinicare-backend-587520969884.us-central1.run.app/health](https://clinicare-backend-587520969884.us-central1.run.app/health) | **Online** ✅ |
| **Container Image** | **Artifact Registry** | `us-central1-docker.pkg.dev/clinicareai/clinicare/backend:latest` | **Active** ✅ |

---

## 🌟 The V5 Multimodal Pipeline

Currently running the **V5 Model (87.6% Out-Of-Fold Accuracy)**, completely transitioning away from legacy heuristics. The pipeline achieves state-of-the-art accuracy by combining three distinct feature extraction modalities:
1. **Vision (PyTorch EfficientNet-B3):** Extracts 1,536 deep latent spatial features from raw handwriting textures.
2. **NLP (EasyOCR):** Extracts 20 linguistic dimensions (medical shorthand frequency, word fragmentation, syntax deterioration).
3. **Handcrafted Heuristics:** Maps 36 structural traits (ink density shifts, quadrant uniformity, stroke edge irregularity).

Extracted features are fused and dimension-reduced via **ANOVA F-value SelectKBest (128 features)** into an aggressively regularized SVM classifier coupled with an interactive XAI (Explainable AI) engine.

---

## 🏗️ Production Architecture & Cloud Deployment

```
Internet / Clients
       │
       ├──► [Firebase Hosting]  clinicare-frontend
       │         • High-speed CDN edge delivery of compiled React 19 SPA
       │         • SSL termination & automated HTTPS
       │         • URL: https://clinicareai.web.app
       │
       └──► [Google Cloud Run]  clinicare-backend
                 • Containerized FastAPI + Uvicorn + CPU PyTorch + EasyOCR + SVM
                 • Auto-scaling (0 → N instances, $0 cost when idle)
                 • Allocated: 4 GiB Memory, 2 vCPU, 300s timeout
                 • URL: https://clinicare-backend-587520969884.us-central1.run.app
```

### 1. Backend Deployment (Google Cloud Run)
The backend is packaged as an optimized, security-hardened Docker container and hosted on Cloud Run.

```bash
# 1. Build & Push Image using Google Cloud Build (from backend directory)
gcloud builds submit \
  --tag us-central1-docker.pkg.dev/clinicareai/clinicare/backend:latest \
  --region=us-central1 \
  --project clinicareai \
  backend/

# 2. Deploy to Cloud Run
gcloud run deploy clinicare-backend \
  --image us-central1-docker.pkg.dev/clinicareai/clinicare/backend:latest \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 4 \
  --min-instances 0 \
  --project clinicareai
```

### 2. Frontend Deployment (Firebase Hosting)
The React 19 + Vite frontend is pre-compiled with the production backend API URL and deployed to Firebase Hosting.

```bash
# 1. Build the production React bundle
cd frontend
$env:VITE_API_URL="https://clinicare-backend-587520969884.us-central1.run.app"  # Windows PowerShell
# export VITE_API_URL="https://clinicare-backend-587520969884.us-central1.run.app" # macOS/Linux
npm run build

# 2. Deploy to Firebase Hosting
firebase deploy --only hosting --project clinicareai
```

---

## 🛡️ Security Audit & Cost Optimization Engineering

To protect user data and ensure Cloud Run compute and storage bills remain minimal:

* **Context Upload Slimming (98% Reduction):** Dedicated `.gcloudignore` and `.dockerignore` files prevent local virtual environments (`.venv/`, `venv/` totaling ~7.4 GB) and raw dataset directories from being uploaded. The Cloud Build tarball is compressed to just **43.6 MiB (16 files)**.
* **CPU-Only PyTorch Optimization:** In the `Dockerfile`, CPU-only PyTorch wheels are installed **before** application dependencies. This prevents `easyocr` and `timm` from pulling 2.5 GB of deadweight NVIDIA CUDA binaries, keeping the final container image size at **1.07 GB** (down from 5.5+ GB).
* **Model Weight Protection:** Wildcard static mounts have been removed. Model weights (`.pth`) and serialized classifiers (`.pkl`) are strictly isolated; attempting to request model files over HTTP returns `404 Not Found`.
* **DoS & Memory Protection:** Added a strict 15 MB payload ceiling (`MAX_UPLOAD_SIZE = 15MB`), rejected empty files, and set a 50-megapixel ceiling (`Image.MAX_IMAGE_PIXELS = 50_000_000`) to block PIL decompression bomb attacks.
* **CORS Hardening:** Replaced wildcard `*` with an explicit origin allowlist (`clinicareai.web.app`, `clinicareai.firebaseapp.com`, and local development ports) with environment variable override support.
* **Security Headers:** Added HTTP middleware enforcing `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and `Referrer-Policy: strict-origin-when-cross-origin`.

---

## 📂 Repository Structure

```
├── backend/
│   ├── data/
│   │   ├── models/                # Production model weights (backbone_v5.pth, burnout_v5_latest.pkl)
│   │   ├── dashboard_data.json    # Aggregated hospital doctor metrics
│   │   ├── dataset.json           # Training dataset metadata
│   │   └── confusion_matrix.png   # Model evaluation confusion matrix
│   ├── Dockerfile                 # Hardened CPU PyTorch container spec
│   ├── .dockerignore              # Backend container exclusion rules
│   ├── .gcloudignore              # Build upload exclusion rules
│   ├── predict.py                 # FastAPI inference server & XAI engine
│   └── requirements-no-torch.txt  # Lightweight Python dependencies
├── frontend/
│   ├── src/                       # React 19 components & pages
│   ├── firebase.json              # Firebase Hosting SPA routing & cache headers
│   ├── .firebaserc                # Firebase project mapping (clinicareai)
│   ├── package.json               # Frontend dependencies & build scripts
│   └── vite.config.js             # Vite configuration with proxy rules
├── ml_pipeline/
│   ├── train_model_v5.py          # MixUp + Consensus Tri-Training pipeline
│   └── generate_cm_plot.py        # Confusion matrix visualizer
├── notebooks/                     # Exploratory Data Analysis & research experiments
├── cloudbuild.yaml                # Google Cloud Build configuration
├── .dockerignore                  # Root container exclusion rules
├── .gcloudignore                  # Root Cloud Build exclusion rules
└── .gitignore                     # Git tracking exclusion rules
```

---

## 🛠️ Local Development Quick Start

Ensure you have **Python 3.10+** and **Node.js 18+** installed on your machine.

### 1. Start the Backend API Locally
```bash
# Create and activate a Virtual Environment
python -m venv venv
.\venv\Scripts\activate   # Windows PowerShell
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Start the Backend Inference Engine
cd backend
python predict.py
```
*(The backend runs at `http://localhost:8000` with automatic device detection for CUDA or CPU).*

### 2. Start the Frontend Application Locally
In a separate terminal, launch the Vite development server:
```bash
cd frontend

# Install Node dependencies
npm install

# Start development server
npm run dev
```
*(The UI runs at `http://localhost:5173` with proxy routing to the local backend).*

---

## 🔬 Explainable AI (XAI)
BurnoutAI isn't a black box. The backend generates interactive feature-weight explanations for every prediction, rendering dynamic insights to the frontend (e.g. *"High clinical shorthand density and irregular stroke edge variation correlated heavily with extreme fatigue"*).

## 📄 License
This system is part of an academic and private research initiative. All rights reserved by the respective authors.
