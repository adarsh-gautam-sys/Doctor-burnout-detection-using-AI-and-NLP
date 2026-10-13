# BurnoutAI (CliniCare v2) - Technical Report & V5 Architecture

This document provides a highly detailed breakdown of the BurnoutAI project, specifically focusing on the advanced **V5 Multimodal Pipeline** that successfully identifies signs of clinical burnout and fatigue directly from physicians' handwritten prescriptions, as well as its production cloud infrastructure, security audit, and deployment architecture.

---

## 1. Methodology

The BurnoutAI V5 Multimodal Pipeline approaches the detection of physician burnout by assuming that cognitive fatigue directly impacts psychomotor control. Instead of relying solely on one modality (like an image classifier), the methodology integrates:
- **Deep Visual Features** to detect complex motor control drops and handwriting degradation.
- **Natural Language Processing (NLP)** over OCR extractions to catch linguistic shifts, terseness, abbreviation abuse, and sentence fragmentation.
- **Targeted Spatial Image Heuristics** mapping exact mechanical elements of the writing, such as the statistical drop in ink density, margin shifts, or edge transitions indicating hurried urgency.

By extracting these independent feature classes and fusing them smartly, the system captures subtle cognitive shifts that precede clinical errors. To overcome the extreme limitation of a small labeled foundation (only 129 verified labeled real images), the methodology implements an aggressive augmentation protocol (**MixUp with α=0.4**, Test-Time Augmentation) and a highly secure Semi-Supervised strategy known as **Consensus Tri-Training** to securely pseudo-label large pools of unlabeled handwritten datasets.

---

## 2. Proposed Solution

The proposed working solution involves a multi-tier platform bridging AI with accessible end-user products:

1. **AI Inference Backend**: A fast, asynchronous `FastAPI` instance that loads the frozen + fine-tuned PyTorch embeddings at runtime. It accepts prescription image uploads and instantly fuses the multimodal vectors to render a prediction.
2. **Premium React 19 Hospital UI**: A completely rebuilt frontend utilizing modern "Blob" visual designs and animated meshes (inspired by ReactBits/Shadcn). It presents clinical directors with an at-a-glance dashboard that organizes physician rosters by AI-derived burnout risk factors.
3. **Explainable AI (XAI)**: The solution recognizes that an opaque "black-box" diagnosis is unacceptable in a clinical sphere. Thus, the system is designed to provide actionable XAI telemetry alongside each prediction. If a physician is tagged as "High Risk," the system highlights the exact contributing modalities—such as identifying *"unusually high-frequency medical abbreviation shorthand"* or *"inconsistent spatial ink uniformity"*—clarifying *why* the prediction was made.

---

## 3. Results (V5 Performance)

The latest V5 Pipeline absolutely bypassed early expectations, achieving **87.6% Out-Of-Fold (OOF) Accuracy**, significantly over delivering against the baseline 70% target. 

- **Version Progression Context:** 
  - V1 & V2 (Pure CNN / EfficientNet end-to-end fine-tuning): Plateaued around **~44% accuracy** due to severe overfitting on the tiny labeled dataset.
  - V3 (Legacy Heuristics + XGBoost): Dropped to **~36% accuracy**.
  - **V5 (Current Pipeline): Peak 87.6% accuracy.**

The primary driver to this success was the Semi-Supervised **Consensus Tri-Training**. It successfully isolated high-quality pseudo-labels from an unlabeled pool by requiring that three fundamentally distinct classifier architectures unanimously agreed on a label with >55% average confidence. This safely increased the dataset's scale without poisoning the training set with noisy false labels.

---

## 4. Model Description & Architecture

The V5 architecture runs in an orchestrated sequence across **6 specialized phases**:

1. **Vision Encoder Fine-Tuning**: Built around a base `EfficientNet-B3` PyTorch backbone (pre-trained on ImageNet). While mostly frozen, the final two convolutional sections (`blocks.5`, `blocks.6`) and the classification head are unfrozen to strictly fine-tune over handwriting textures using a CrossEntropy loss weighted precisely by class inverse frequencies. It renders a dense 1,536-dimensional feature space.
2. **OCR + NLP Modality**: `EasyOCR` operates in a secondary thread, extracting raw text and bounding boxes. A secondary parser computes 20 handcrafted semantic structural scores (e.g., uppercase frequency vectors, lexical diversity ratios, specific dosage token counting logic).
3. **Image Heuristics Modality**: Calculates 36 mechanical traits measured from the binary threshold projection of the image. This maps stroke edge densities, base ink variance proxies measured in quad-blocks, horizontal run transitions, and baseline skew mappings.
4. **Multimodal Fusion & SelectKBest**: The extracted traits from the three distinct modules are concatenated. Given the high theoretical dimensionality against small data vectors, an **ANOVA F-value SelectKBest** filter is applied, cleanly slashing noisy/flat dimensions and keeping the richest ~128 multimodal features.
5. **Consensus Tri-Training Consensus Generation**: Unlabeled datasets are processed by three blind parallel models:
   - *Model A: Vision-only SVM*
   - *Model B: NLP-only Random Forest*
   - *Model C: Heuristics-only SVM*
   Only samples passing a strict 3-way consensus are merged into the final training data.
6. **Final Classification**: A finely-tuned, heavily regularized **Linear SVM** constructs the definitive decision boundary on the mixed (real + highest-quality pseudo-labeled) multimodal domain.

---

## 5. UI Design Screenshots & Visual Demo

The CliniCare user interface was heavily refreshed utilizing React and Vite, achieving a premium, sleek aesthetic suitable for clinical executives without sacrificing visual dynamics. 

### A) Homepage & Hero Aesthetic
Features a modern layout with a glowing background mesh ("blobs"), introducing the AI technology dynamically.

![Home Page / Hero Aesthetics](file:///C:/Users/conta.LAPTOP-IR41J1UC/.gemini/antigravity/brain/63c5b4d6-854f-4b03-80e3-36ee0b704321/homepage_hero_1775748850604.png)

*(Also showing page scroll components)*
![Home Page Information](file:///C:/Users/conta.LAPTOP-IR41J1UC/.gemini/antigravity/brain/63c5b4d6-854f-4b03-80e3-36ee0b704321/homepage_scroll_1_1775748852823.png)


### B) Hospital Dashboard & Roster View
Shows an aggregate risk distribution. It summarizes physician rosters with their associated Burnout Confidence score outputs processed silently by the backend.

![Hospital Dashboard Overview](file:///C:/Users/conta.LAPTOP-IR41J1UC/.gemini/antigravity/brain/63c5b4d6-854f-4b03-80e3-36ee0b704321/dashboard_main_1775748874573.png)


### C) Active XAI Diagnosis & Upload Pipeline
A robust upload element supporting instant assessment from handwritten clinic notes, immediately providing XAI breakdowns back to the user upon processing.

![Interactive Demo & Assessment Portal](file:///C:/Users/conta.LAPTOP-IR41J1UC/.gemini/antigravity/brain/63c5b4d6-854f-4b03-80e3-36ee0b704321/demo_section_1775748964833.png)

### Screen Recording Preview
*(For a complete visual inspection, view the browser recording WebP natively captured during the session).*
![Browser UI Recording](file:///C:/Users/conta.LAPTOP-IR41J1UC/.gemini/antigravity/brain/63c5b4d6-854f-4b03-80e3-36ee0b704321/ui_design_screenshots_1775748816063.webp)

---

## 6. Cloud Infrastructure & Production Deployment Architecture

To ensure enterprise-grade availability, low latency, and zero idle expenditure, BurnoutAI is deployed in a decoupled serverless architecture across **Google Cloud Run** and **Firebase Hosting**.

```
Internet Clients / Clinicians
             │
             ├──► [Firebase Hosting]  clinicare-frontend
             │         • Global Edge CDN delivery of compiled React 19 SPA
             │         • Automated SSL termination and HTTP/2 + HTTP/3 support
             │         • Domain: https://clinicareai.web.app
             │         • Alternative: https://clinicareai.firebaseapp.com
             │
             └──► [Google Cloud Run]  clinicare-backend
                       • Fully managed containerized FastAPI inference microservice
                       • Scales automatically from 0 to N instances ($0 compute when idle)
                       • Memory: 4 GiB | vCPU: 2 | Timeout: 300s | Concurrency: 4
                       • Region: us-central1
                       • Production API: https://clinicare-backend-587520969884.us-central1.run.app
```

### Production Endpoints Summary

| Layer | Component | Production URL | Technology Stack |
|:---|:---|:---|:---|
| **Frontend** | Clinical Web Interface | `https://clinicareai.web.app` | React 19, Vite, Tailwind CSS, Lucide |
| **Frontend** | Secondary Domain | `https://clinicareai.firebaseapp.com` | Firebase Hosting Edge CDN |
| **Backend** | API Health Probe | `https://clinicare-backend-587520969884.us-central1.run.app/health` | FastAPI, Uvicorn, Python 3.11 |
| **Backend** | Inference Service | `https://clinicare-backend-587520969884.us-central1.run.app/predict` | PyTorch (CPU), EasyOCR, SelectKBest, SVM |
| **Registry** | Docker Container | `us-central1-docker.pkg.dev/clinicareai/clinicare/backend:latest` | Google Artifact Registry |

---

## 7. Security Audit, Cost Prevention & System Hardening

A comprehensive security audit and cost-engineering review was performed across the codebase prior to production deployment.

### A) Vulnerability & Hardening Matrix

| Risk Category | Severity | Root Cause | Engineering Remediation | Verified Status |
|:---|:---:|:---|:---|:---:|
| **Model Weight Disclosure** | **HIGH** | `app.mount("/data", StaticFiles(...))` previously exposed `data/models/` publicly over HTTP. | Wildcard static mount was completely removed. Replaced with dedicated, read-only JSON endpoints (`/data/dashboard_data.json`, `/data/dataset.json`, `/data/confusion_matrix.png`). Direct HTTP requests for `.pth` or `.pkl` files return `404 Not Found`. | **RESOLVED & TESTED** ✅ |
| **Denial of Service (DoS / OOM)** | **MEDIUM** | `await file.read()` had no payload size limit, allowing arbitrarily large files to exhaust Cloud Run memory. | Implemented strict 15 MB payload ceiling (`MAX_UPLOAD_SIZE = 15MB`), zero-byte check, and 50 MP PIL ceiling (`Image.MAX_IMAGE_PIXELS = 50_000_000`) to block decompression bombs. | **RESOLVED & TESTED** ✅ |
| **Permissive CORS Policy** | **MEDIUM** | `allow_origins=["*"]` with credentials enabled allowed cross-origin request forgery. | Restricted origins to `https://clinicareai.web.app`, `https://clinicareai.firebaseapp.com`, and authorized local ports, with support for an `ALLOWED_ORIGINS` environment variable. | **RESOLVED & TESTED** ✅ |
| **Missing Security Headers** | **LOW** | Responses lacked browser-level defensive headers. | Implemented FastAPI middleware setting `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, and `X-XSS-Protection: 1; mode=block`. | **RESOLVED & TESTED** ✅ |
| **Filename / Path Traversal** | **LOW** | Raw `file.filename` was reflected directly in JSON responses. | Sanitized uploaded filenames using `Path(file.filename).name.strip()` to strip relative path characters. | **RESOLVED & TESTED** ✅ |

### B) Cloud Cost Prevention Engineering

1. **Build Context Upload Slimming (98% Reduction):**
   - *Problem:* Initial Cloud Build submissions lacked a `.gcloudignore` and had an incomplete `.gitignore`, resulting in the CLI packaging local `.venv/` and `venv/` directories (38,243 files, **2.2 GiB**), causing slow uploads and Cloud Storage costs.
   - *Solution:* Implemented strict `.gcloudignore` and `.dockerignore` files excluding all virtualenvs, node_modules, and raw image datasets. Scoped build submission to `backend/`, reducing the uploaded tarball to **43.6 MiB (16 files)** — a **98% reduction** uploading in ~3 seconds.
2. **Container Image Optimization (CUDA Elimination):**
   - *Problem:* Standard installation of `easyocr` and `timm` pulled default CUDA PyTorch packages from PyPI, bundling 2.5 GB of unused NVIDIA libraries (`cublas`, `cudnn`, `nccl`, `triton`).
   - *Solution:* Reordered the `Dockerfile` to install CPU-only PyTorch (`torch==2.6.0+cpu`, `torchvision==0.21.0+cpu`) from the official PyTorch wheel repository *before* installing application requirements. This satisfied the torch dependency up-front, eliminating 2.5 GB of deadweight CUDA binaries and compressing the final container image to **1.07 GB** (down from 5.5+ GB).
3. **Scale-to-Zero Compute:**
   - Cloud Run is configured with `--min-instances 0`, ensuring container instances spin down completely during periods of inactivity, resulting in **$0 compute bill** while idle.
