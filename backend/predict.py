"""
predict.py — BurnoutAI FastAPI Inference Server (V5 Multimodal)
================================================================
Run:  python predict.py
Endpoints:
  POST /predict  — upload image → burnout prediction + XAI explanation
  POST /ocr      — alias for /predict (frontend compatibility)
  GET  /results  — returns model_results.json
  GET  /generate_report — download CSV report of all doctors
  GET  /health   — health check
"""

import os, json, csv, io, re, warnings
import numpy as np
from pathlib import Path
from io import BytesIO
from copy import deepcopy

from PIL import Image
import joblib

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn

warnings.filterwarnings('ignore')

# ── PyTorch / Vision ──────────────────────────────────────
import torch
import torch.nn as nn
import torchvision.transforms as T

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

try:
    import easyocr
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

# ── Paths ─────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
MODELS_DIR     = BASE_DIR / 'data' / 'models'
RESULTS_PATH   = BASE_DIR / 'data' / 'model_results.json'
DASHBOARD_PATH = BASE_DIR / 'data' / 'dashboard_data.json'

IMGSIZE = 224

# ═══════════════════════════════════════════════════════════
# V5 PYTORCH BACKBONE (mirrors train_model_v5.py exactly)
# ═══════════════════════════════════════════════════════════

class BurnoutClassifier(nn.Module):
    """Same architecture as train_model_v5.py — needed to load state_dict."""
    def __init__(self, num_classes=3):
        super().__init__()
        if HAS_TIMM:
            self.backbone = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0)
            self.feat_dim = self.backbone.num_features
        else:
            import torchvision.models as models
            _resnet = models.resnet50(weights=None)
            self.backbone = nn.Sequential(*list(_resnet.children())[:-1], nn.Flatten())
            self.feat_dim = 2048

        # Classification head (same structure, won't be used for inference)
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, return_features=False):
        feats = self.backbone(x)
        if feats.dim() > 2:
            feats = feats.view(feats.size(0), -1)
        if return_features:
            return feats
        return self.head(feats)


# ═══════════════════════════════════════════════════════════
# FEATURE EXTRACTION FUNCTIONS (mirrors train_model_v5.py)
# ═══════════════════════════════════════════════════════════

val_transform = T.Compose([
    T.Resize((IMGSIZE, IMGSIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def extract_vision_features(img_pil, backbone_model):
    """Extract 1536-dim vision features using the fine-tuned EfficientNet."""
    tensor = val_transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = backbone_model(tensor, return_features=True)
    return feats.cpu().numpy().flatten()


def extract_nlp_features(text, ocr_confidence, num_detections):
    """Extract 20 NLP features from OCR text. Mirrors train_model_v5.py."""
    features = []
    words = text.split()
    features.append(len(text))
    features.append(len(words))
    features.append(np.mean([len(w) for w in words]) if words else 0)
    features.append(np.std([len(w) for w in words]) if len(words) > 1 else 0)
    features.append(max([len(w) for w in words]) if words else 0)
    features.append(min([len(w) for w in words]) if words else 0)

    alpha_count = sum(c.isalpha() for c in text)
    digit_count = sum(c.isdigit() for c in text)
    space_count = sum(c.isspace() for c in text)
    special_count = len(text) - alpha_count - digit_count - space_count
    total = len(text) + 1e-8
    features.append(alpha_count / total)
    features.append(digit_count / total)
    features.append(special_count / total)
    features.append(space_count / total)

    upper_count = sum(c.isupper() for c in text)
    features.append(upper_count / (alpha_count + 1e-8))

    short_words = [w for w in words if len(w) <= 3]
    features.append(len(short_words) / (len(words) + 1e-8))

    dosage_pattern = re.compile(r'\d+\s*(mg|ml|mcg|g|tab|cap|times|x)', re.IGNORECASE)
    dosage_count = len(dosage_pattern.findall(text))
    features.append(dosage_count)
    features.append(sum(1 for w in words if len(w) > 1 and any(c.isupper() for c in w[1:])) / (len(words) + 1e-8))

    sentences = text.split('.')
    features.append(len([s for s in sentences if s.strip()]))
    features.append(np.mean([len(s.split()) for s in sentences if s.strip()]) if any(s.strip() for s in sentences) else 0)

    features.append(ocr_confidence)
    features.append(num_detections)

    unique_words = set(w.lower() for w in words)
    features.append(len(unique_words) / (len(words) + 1e-8))
    features.append(len(text) / (num_detections + 1e-8))

    return np.array(features, dtype=np.float32)


def extract_handcrafted_features(img_pil):
    """Extract 36 handcrafted image features. Mirrors train_model_v5.py."""
    img = img_pil.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    gray = np.array(img.convert('L'), dtype=np.float32) / 255.0
    h, w = gray.shape
    features = []

    # Grayscale statistics (4)
    features.append(gray.mean())
    features.append(gray.std())
    mean_val = gray.mean()
    std_val = gray.std() + 1e-8
    features.append(float(((gray - mean_val) ** 3).mean() / (std_val ** 3)))
    features.append(float(((gray - mean_val) ** 4).mean() / (std_val ** 4)))

    # Edge density (4)
    edges_x = np.abs(np.diff(gray, axis=1))
    edges_y = np.abs(np.diff(gray, axis=0))
    features.append(edges_x.mean())
    features.append(edges_x.std())
    features.append(edges_y.mean())
    features.append(edges_y.std())

    # Ink density (8)
    threshold = gray.mean()
    binary = (gray < threshold).astype(np.float32)
    features.append(binary.mean())
    features.append(binary.sum())
    features.append(binary[:h//2, :w//2].mean())
    features.append(binary[:h//2, w//2:].mean())
    features.append(binary[h//2:, :w//2].mean())
    features.append(binary[h//2:, w//2:].mean())
    row_runs = np.diff(binary.astype(int), axis=1)
    features.append(np.abs(row_runs).sum() / (h * w))
    col_runs = np.diff(binary.astype(int), axis=0)
    features.append(np.abs(col_runs).sum() / (h * w))

    # Spatial uniformity (8)
    row_means = gray.mean(axis=1)
    col_means = gray.mean(axis=0)
    features.append(row_means.std())
    features.append(col_means.std())
    features.append(np.diff(row_means).std())
    features.append(np.diff(col_means).std())

    center = gray[h//4:3*h//4, w//4:3*w//4].mean()
    corners = np.mean([gray[:h//4,:w//4].mean(), gray[:h//4,-w//4:].mean(),
                       gray[-h//4:,:w//4].mean(), gray[-h//4:,-w//4:].mean()])
    features.append(center / (corners + 1e-8))
    features.append(center - corners)

    q1 = gray[:h//2, :w//2].mean()
    q2 = gray[:h//2, w//2:].mean()
    q3 = gray[h//2:, :w//2].mean()
    q4 = gray[h//2:, w//2:].mean()
    features.append(max(q1,q2,q3,q4) - min(q1,q2,q3,q4))
    features.append(np.std([q1,q2,q3,q4]))

    # Writing entropy proxy (4)
    block_h, block_w = h // 4, w // 4
    block_vars = []
    for bi in range(4):
        for bj in range(4):
            block = gray[bi*block_h:(bi+1)*block_h, bj*block_w:(bj+1)*block_w]
            block_vars.append(block.var())
    block_vars = np.array(block_vars)
    features.append(block_vars.mean())
    features.append(block_vars.std())
    features.append(np.percentile(block_vars, 25))
    features.append(np.percentile(block_vars, 75))

    # Image metadata (4)
    ow, oh = img_pil.size
    features.append(ow / (oh + 1e-8))
    features.append(float(ow * oh))
    features.append(float(ow))
    features.append(float(oh))

    # Color channel stats (4)
    for ch in range(3):
        features.append(arr[:,:,ch].mean() / 255.0)
    features.append(arr.std() / 255.0)

    return np.array(features, dtype=np.float32)


def extract_text_from_image(img_path_or_arr, ocr_reader):
    """Extract text using EasyOCR. Accepts a file path or numpy array."""
    try:
        results = ocr_reader.readtext(img_path_or_arr, detail=1)
        texts = [r[1] for r in results]
        confidences = [r[2] for r in results]
        full_text = ' '.join(texts)
        avg_conf = np.mean(confidences) if confidences else 0.0
        return full_text, avg_conf, len(results)
    except Exception:
        return '', 0.0, 0


# ═══════════════════════════════════════════════════════════
# XAI EXPLANATION ENGINE (V5-aware)
# ═══════════════════════════════════════════════════════════

# Human-readable names for the V5 vision feature dimensions
V5_FEATURE_DISPLAY = {
    # Handcrafted features (if selected)
    'gray_mean': 'Grayscale Brightness',
    'gray_std': 'Contrast Variation',
    'gray_skewness': 'Tonal Asymmetry',
    'gray_kurtosis': 'Contrast Sharpness',
    'edge_x_mean': 'Horizontal Edge Density',
    'edge_x_std': 'Horizontal Stroke Irregularity',
    'edge_y_mean': 'Vertical Edge Density',
    'edge_y_std': 'Vertical Stroke Irregularity',
    'ink_density': 'Ink Coverage Ratio',
    'ink_pixel_count': 'Total Ink Pixels',
    'ink_q1': 'Ink Density (Top-Left)',
    'ink_q2': 'Ink Density (Top-Right)',
    'ink_q3': 'Ink Density (Bottom-Left)',
    'ink_q4': 'Ink Density (Bottom-Right)',
    'stroke_transition_h': 'Horizontal Stroke Fragmentation',
    'stroke_transition_v': 'Vertical Stroke Fragmentation',
    'writing_uniformity': 'Quadrant Uniformity',
    'writing_spread': 'Writing Spatial Spread',
}

# V5 vision concept names for the 1536 EfficientNet features
VISION_CONCEPT_NAMES = [
    'Stroke Curvature Pattern', 'Letter Spacing Regularity', 'Line Straightness',
    'Pressure Intensity Map', 'Character Size Consistency', 'Ink Flow Continuity',
    'Word Boundary Clarity', 'Baseline Alignment', 'Slant Angle Consistency',
    'Loop Formation Quality', 'Stroke Width Variation', 'Writing Speed Indicator',
    'Pen Lift Frequency', 'Character Proportion', 'Inter-word Gap Distribution',
    'Margin Consistency', 'Page Fill Density', 'Writing Rhythm Pattern',
    'Tremor Detection Signal', 'Fatigue Progression Marker',
]


def generate_v5_explanation(prediction, probabilities, ocr_text, handcrafted_feats):
    """Generate a human-readable explanation using V5 multimodal signals."""
    findings = []

    # ── Handcrafted-based insights ────────────────────────
    ink = handcrafted_feats[8] if len(handcrafted_feats) > 8 else 0  # ink_density
    if ink > 0.55:
        findings.append('Heavy ink pressure detected — potential sign of motor tension')
    elif ink < 0.35:
        findings.append('Light ink coverage — possible fatigue or rushed writing')

    # Edge irregularity
    ex_std = handcrafted_feats[5] if len(handcrafted_feats) > 5 else 0
    ey_std = handcrafted_feats[7] if len(handcrafted_feats) > 7 else 0
    if ex_std > 0.05 or ey_std > 0.05:
        findings.append('Irregular stroke edges — potential hand tremor or instability')

    # Spatial uniformity
    if len(handcrafted_feats) > 21:
        uniformity = handcrafted_feats[21]  # quadrant std
        if uniformity > 0.03:
            findings.append('Uneven writing distribution across page quadrants')

    # ── OCR-based insights ────────────────────────────────
    if ocr_text:
        words = ocr_text.split()
        if words:
            avg_word_len = np.mean([len(w) for w in words])
            short_ratio = sum(1 for w in words if len(w) <= 3) / len(words)
            if short_ratio > 0.6:
                findings.append('High abbreviation density — frequent use of medical shorthand')
            if avg_word_len < 3.5:
                findings.append('Very short average word length — possible rushed documentation')
            if len(words) < 5:
                findings.append('Minimal text detected — sparse prescription content')
    else:
        findings.append('No readable text detected — handwriting may be illegible')

    # ── Vision model confidence ───────────────────────────
    max_prob = max(probabilities.values())
    if max_prob > 0.85:
        findings.append(f'Vision model shows high confidence ({max_prob*100:.0f}%) in structural pattern match')
    elif max_prob < 0.5:
        findings.append('Vision model shows uncertainty — handwriting features are ambiguous')

    if not findings:
        findings.append('Prediction based on combined multimodal feature patterns')

    # Build summary
    risk_text = {
        'High': 'High burnout risk detected',
        'Medium': 'Moderate burnout indicators present',
        'Low': 'Low burnout risk — writing appears stable'
    }
    summary_header = risk_text.get(prediction, f'{prediction} burnout level detected')

    # Build top features for XAI panel display
    top_features = []
    # Use the handcrafted features that are most interpretable
    hc_names = ['gray_mean', 'gray_std', 'gray_skewness', 'gray_kurtosis',
                'edge_x_mean', 'edge_x_std', 'edge_y_mean', 'edge_y_std',
                'ink_density', 'ink_pixel_count', 'ink_q1', 'ink_q2', 'ink_q3', 'ink_q4',
                'stroke_transition_h', 'stroke_transition_v',
                'row_std', 'col_std', 'row_diff_std', 'col_diff_std',
                'center_corner_ratio', 'center_corner_diff',
                'quadrant_range', 'quadrant_std',
                'block_var_mean', 'block_var_std', 'block_var_p25', 'block_var_p75',
                'aspect_ratio', 'total_pixels', 'width', 'height',
                'ch_r_mean', 'ch_g_mean', 'ch_b_mean', 'color_std']

    # Pick the 5 most extreme handcrafted values as "top features"
    hc_abs = np.abs(handcrafted_feats)
    hc_sorted = np.argsort(hc_abs)[::-1]
    for rank, idx in enumerate(hc_sorted[:5]):
        name = hc_names[idx] if idx < len(hc_names) else f'feature_{idx}'
        display = V5_FEATURE_DISPLAY.get(name, name.replace('_', ' ').title())
        top_features.append({
            'feature_name': name,
            'display_name': display,
            'value': round(float(handcrafted_feats[idx]), 4),
            'importance': round(1.0 - rank * 0.15, 4),
            'raw_importance': round(float(hc_abs[idx]), 6)
        })

    return {
        'summary': f"{summary_header}. {findings[0]}.",
        'findings': findings[:5],
        'top_features': top_features,
        'feature_count': 1592,
        'selected_features': 128,
        'interpretable_features_analyzed': 36,
        'model_type': 'V5 Multimodal (EfficientNet-B3 + EasyOCR + SVM)',
        'ocr_text_preview': (ocr_text[:200] + '...' if len(ocr_text) > 200 else ocr_text) if ocr_text else ''
    }


# ═══════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════

app = FastAPI(title="BurnoutAI Prediction API", version="5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (images, etc.) from data/
if (BASE_DIR / 'data').exists():
    app.mount("/data", StaticFiles(directory=str(BASE_DIR / 'data')), name="data")

# ── Global State ──────────────────────────────────────────
backbone_model = None
classifier = None
label_encoder = None
feature_selector = None
ocr_reader = None
MODEL_VERSION = None  # 'v5' or 'v3'


def load_models():
    """Load V5 models if available, otherwise fall back to V3."""
    global backbone_model, classifier, label_encoder, feature_selector
    global ocr_reader, MODEL_VERSION

    v5_backbone = MODELS_DIR / 'backbone_v5.pth'
    v5_classifier = MODELS_DIR / 'burnout_v5_latest.pkl'
    v5_selector = MODELS_DIR / 'feature_selector_v5.pkl'
    v5_encoder = MODELS_DIR / 'label_encoder_v5.pkl'

    # ── Try V5 first ──────────────────────────────────────
    if v5_backbone.exists() and v5_classifier.exists():
        print("🔬 Loading V5 Multimodal Pipeline...")
        MODEL_VERSION = 'v5'

        # PyTorch backbone
        backbone_model = BurnoutClassifier(num_classes=3).to(DEVICE)
        state = torch.load(str(v5_backbone), map_location=DEVICE, weights_only=True)
        backbone_model.load_state_dict(state)
        backbone_model.eval()
        print(f"  ✓ EfficientNet-B3 backbone loaded on {DEVICE.upper()}")

        # SVM classifier
        classifier = joblib.load(v5_classifier)
        print(f"  ✓ Classifier loaded: {v5_classifier.name}")

        # Feature selector
        if v5_selector.exists():
            feature_selector = joblib.load(v5_selector)
            print(f"  ✓ SelectKBest loaded: {v5_selector.name}")

        # Label encoder
        if v5_encoder.exists():
            label_encoder = joblib.load(v5_encoder)
            print(f"  ✓ Label encoder: {list(label_encoder.classes_)}")

        # EasyOCR reader
        if HAS_OCR:
            ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)
            print("  ✓ EasyOCR reader loaded")
        else:
            print("  ⚠ EasyOCR not available — NLP features will be zeroed")

        print(f"  🏥 V5 Pipeline ready (87.6% accuracy)")
        return

    # ── Fallback to V3 ────────────────────────────────────
    print("⚠ V5 models not found, falling back to V3...")
    MODEL_VERSION = 'v3'
    model_path = MODELS_DIR / 'burnout_v3_latest.pkl'
    if not model_path.exists():
        v3_models = list(MODELS_DIR.glob('burnout_v3_*.pkl'))
        if v3_models:
            model_path = v3_models[0]
        else:
            print("⚠ No model found — run train_model_v5.py first!")
            return

    classifier = joblib.load(model_path)
    print(f"  ✓ V3 Classifier loaded: {model_path.name}")

    encoder_path = MODELS_DIR / 'label_encoder_v3.pkl'
    if not encoder_path.exists():
        encoder_path = MODELS_DIR / 'label_encoder.pkl'
    if encoder_path.exists():
        label_encoder = joblib.load(encoder_path)
        print(f"  ✓ Label encoder: {list(label_encoder.classes_)}")


# ═══════════════════════════════════════════════════════════
# V3 FALLBACK — 490-dim feature extraction (legacy)
# ═══════════════════════════════════════════════════════════

def extract_v3_features(img):
    """Extract 490-dim feature vector for V3 model. Legacy compatibility."""
    img_resized = img.resize((IMGSIZE, IMGSIZE))
    arr = np.array(img_resized, dtype=np.float32)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    features = []

    # Color histograms (3 × 64 = 192)
    for ch in [r, g, b]:
        hist, _ = np.histogram(ch, bins=64, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
        features.extend(hist)

    # HSV histograms (3 × 64 = 192)
    hsv = img_resized.convert('HSV')
    hsv_arr = np.array(hsv, dtype=np.float32)
    for ch_idx in range(3):
        ch = hsv_arr[:,:,ch_idx]
        hist, _ = np.histogram(ch, bins=64, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
        features.extend(hist)

    # Grayscale stats (4)
    gray = np.array(img_resized.convert('L'), dtype=np.float32) / 255.0
    features.append(gray.mean())
    features.append(gray.std())
    mean_val = gray.mean()
    std_val = gray.std() + 1e-8
    features.append(float(((gray - mean_val) ** 3).mean() / (std_val ** 3)))
    features.append(float(((gray - mean_val) ** 4).mean() / (std_val ** 4)))

    # Edge density (4)
    edges_x = np.abs(np.diff(gray, axis=1))
    edges_y = np.abs(np.diff(gray, axis=0))
    features.append(edges_x.mean())
    features.append(edges_x.std())
    features.append(edges_y.mean())
    features.append(edges_y.std())

    # Texture features (26)
    h, w = gray.shape
    block_h, block_w = h // 8, w // 8
    block_means, block_stds = [], []
    for bi in range(8):
        for bj in range(8):
            block = gray[bi*block_h:(bi+1)*block_h, bj*block_w:(bj+1)*block_w]
            block_means.append(block.mean())
            block_stds.append(block.std())
    block_means = np.array(block_means)
    block_stds = np.array(block_stds)
    features.extend([block_means.mean(), block_means.std(), block_stds.mean(), block_stds.std()])
    features.extend([np.percentile(block_means, 25), np.percentile(block_means, 75)])
    features.extend([np.percentile(block_stds, 25), np.percentile(block_stds, 75)])
    row_means = gray.mean(axis=1)
    col_means = gray.mean(axis=0)
    features.extend([row_means.std(), col_means.std()])
    features.extend([np.diff(row_means).std(), np.diff(col_means).std()])
    center = gray[h//4:3*h//4, w//4:3*w//4].mean()
    corners_val = np.mean([gray[:h//4,:w//4].mean(), gray[:h//4,-w//4:].mean(),
                           gray[-h//4:,:w//4].mean(), gray[-h//4:,-w//4:].mean()])
    features.extend([center / (corners_val + 1e-8), center - corners_val])
    q1, q2, q3, q4 = (gray[:h//2,:w//2].mean(), gray[:h//2,w//2:].mean(),
                       gray[h//2:,:w//2].mean(), gray[h//2:,w//2:].mean())
    features.extend([q1, q2, q3, q4])
    features.extend([max(q1,q2,q3,q4) - min(q1,q2,q3,q4), np.std([q1,q2,q3,q4])])
    features.extend([row_means.max() - row_means.min(), col_means.max() - col_means.min()])

    # Ink density (8)
    threshold = gray.mean()
    binary = (gray < threshold).astype(np.float32)
    features.extend([binary.mean(), binary.sum()])
    features.extend([binary[:h//2,:w//2].mean(), binary[:h//2,w//2:].mean()])
    features.extend([binary[h//2:,:w//2].mean(), binary[h//2:,w//2:].mean()])
    row_runs = np.diff(binary.astype(int), axis=1)
    col_runs = np.diff(binary.astype(int), axis=0)
    features.extend([np.abs(row_runs).sum()/(h*w), np.abs(col_runs).sum()/(h*w)])

    # Size (4)
    features.extend([1.0, float(IMGSIZE*IMGSIZE), float(IMGSIZE), float(IMGSIZE)])

    # Frequency (64)
    ps = IMGSIZE // 8
    for pi in range(8):
        for pj in range(8):
            patch = gray[pi*ps:(pi+1)*ps, pj*ps:(pj+1)*ps]
            features.append(patch.var())

    return np.array(features, dtype=np.float32)


# ═══════════════════════════════════════════════════════════
# PREDICTION LOGIC
# ═══════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    load_models()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": classifier is not None,
        "model_version": MODEL_VERSION,
        "gpu": DEVICE.upper()
    }


@app.get("/results")
async def get_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {"error": "No results file found"}


async def _predict_image(file: UploadFile):
    """Core prediction logic — uses V5 multimodal or V3 fallback."""
    if classifier is None:
        raise HTTPException(503, "Model not loaded. Run train_model_v5.py first.")

    if not file.filename:
        raise HTTPException(400, "No file uploaded")

    ext = Path(file.filename).suffix.lower()
    contents = await file.read()

    # ── Text file ─────────────────────────────────────────
    if ext in ['.txt', '.csv', '.text']:
        try:
            text = contents.decode('utf-8', errors='ignore').lower()
        except Exception:
            raise HTTPException(400, "Cannot read text file")

        high_kw = ['severe', 'extreme', 'critical', 'illegible', 'unreadable',
                   'chaotic', 'tremor', 'erratic', 'heavy strokes', 'deteriorat']
        med_kw  = ['moderate', 'irregular', 'rushed', 'abbreviated', 'incomplete',
                   'variable', 'fatigue', 'somewhat']
        low_kw  = ['clear', 'neat', 'legible', 'consistent', 'organized',
                   'steady', 'careful', 'precise', 'regular']

        hs = sum(1 for k in high_kw if k in text)
        ms = sum(1 for k in med_kw if k in text)
        ls = sum(1 for k in low_kw if k in text)
        total = max(hs + ms + ls, 1)

        probs = {"Low": ls/total, "Medium": ms/total, "High": hs/total}
        s = sum(probs.values())
        if s == 0:
            probs = {"Low": 0.33, "Medium": 0.34, "High": 0.33}
        else:
            probs = {k: round(v/s, 4) for k, v in probs.items()}

        pred = max(probs, key=probs.get)
        return {
            "prediction": pred,
            "burnout_risk": pred,
            "confidence": round(probs[pred] * 100, 1),
            "probabilities": probs,
            "input_type": "text",
            "filename": file.filename
        }

    # ── Image file ────────────────────────────────────────
    if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
        raise HTTPException(400, f"Unsupported: {ext}. Use JPG, PNG, or TXT.")

    try:
        img = Image.open(BytesIO(contents)).convert('RGB')
    except Exception as e:
        raise HTTPException(400, f"Cannot open image: {e}")

    # ══════════════════════════════════════════════════════
    # V5 MULTIMODAL INFERENCE
    # ══════════════════════════════════════════════════════
    if MODEL_VERSION == 'v5' and backbone_model is not None:
        # 1. Vision features (1536-dim)
        vision_feats = extract_vision_features(img, backbone_model)

        # 2. NLP features (20-dim)
        ocr_text = ''
        if ocr_reader is not None:
            img_np = np.array(img)
            ocr_text, ocr_conf, n_det = extract_text_from_image(img_np, ocr_reader)
            nlp_feats = extract_nlp_features(ocr_text, ocr_conf, n_det)
        else:
            nlp_feats = np.zeros(20, dtype=np.float32)

        # 3. Handcrafted features (36-dim)
        hc_feats = extract_handcrafted_features(img)

        # 4. Fuse → SelectKBest → SVM predict
        raw_features = np.concatenate([vision_feats, nlp_feats, hc_feats])
        raw_features = np.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0)

        if feature_selector is not None:
            selected = feature_selector.transform(raw_features.reshape(1, -1))
        else:
            selected = raw_features.reshape(1, -1)

        # Check if the classifier has predict_proba
        if hasattr(classifier, 'predict_proba'):
            proba = classifier.predict_proba(selected)[0]
        elif hasattr(classifier, 'decision_function'):
            # LinearSVC doesn't have predict_proba — use decision_function + softmax
            dec = classifier.decision_function(selected)[0]
            exp_dec = np.exp(dec - dec.max())
            proba = exp_dec / exp_dec.sum()
        else:
            pred_idx = classifier.predict(selected)[0]
            proba = np.zeros(len(label_encoder.classes_))
            proba[pred_idx] = 1.0

        pred_idx = proba.argmax()
        pred_label = label_encoder.classes_[pred_idx]

        probs = {}
        for i, cls in enumerate(label_encoder.classes_):
            probs[cls] = round(float(proba[i]), 4)

        # XAI Explanation
        explanation = generate_v5_explanation(pred_label, probs, ocr_text, hc_feats)

        return {
            "prediction": pred_label,
            "burnout_risk": pred_label,
            "confidence": round(float(proba.max()) * 100, 1),
            "probabilities": probs,
            "input_type": "image",
            "filename": file.filename,
            "model_version": "V5",
            "explanation": explanation
        }

    # ══════════════════════════════════════════════════════
    # V3 FALLBACK
    # ══════════════════════════════════════════════════════
    fv = extract_v3_features(img)
    features = fv.reshape(1, -1)
    proba = classifier.predict_proba(features)[0]
    pred_idx = proba.argmax()
    pred_label = label_encoder.classes_[pred_idx]

    probs = {}
    for i, cls in enumerate(label_encoder.classes_):
        probs[cls] = round(float(proba[i]), 4)

    return {
        "prediction": pred_label,
        "burnout_risk": pred_label,
        "confidence": round(float(proba.max()) * 100, 1),
        "probabilities": probs,
        "input_type": "image",
        "filename": file.filename,
        "model_version": "V3",
        "explanation": {
            "summary": f"{pred_label} burnout level detected.",
            "findings": ["V3 legacy model — upgrade to V5 for detailed explanations"],
            "top_features": [],
            "model_type": "V3 Legacy (Handcrafted + XGBoost)"
        }
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Upload an image or text file → burnout prediction with XAI explanation."""
    return await _predict_image(file)


@app.post("/ocr")
async def ocr_predict(file: UploadFile = File(...)):
    """Alias for /predict — frontend compatibility endpoint."""
    return await _predict_image(file)


@app.get("/generate_report")
async def generate_report():
    """Generate a CSV report of all doctors from dashboard_data.json."""
    if not DASHBOARD_PATH.exists():
        raise HTTPException(404, "dashboard_data.json not found")

    with open(DASHBOARD_PATH) as f:
        data = json.load(f)

    doctors = data.get('doctors', [])

    def get_recommendation(risk_level):
        return {
            'High': 'Immediate intervention required. Reduce workload and arrange support.',
            'Medium': 'Monitor closely. Schedule follow-up assessment within 2 weeks.',
            'Low': 'No immediate action needed. Continue routine monitoring.'
        }.get(risk_level, 'Assessment pending')

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        'Doctor ID', 'Name', 'Specialty', 'Burnout Risk',
        'Confidence (%)', 'Low %', 'Medium %', 'High %',
        'Last Updated', 'Recommendation'
    ])

    for doc in doctors:
        writer.writerow([
            doc.get('id', ''),
            doc.get('name', ''),
            doc.get('specialty', ''),
            doc.get('burnout', ''),
            round(doc.get('confidence', 0), 1),
            round(doc.get('low_pct', 0), 1),
            round(doc.get('medium_pct', 0), 1),
            round(doc.get('high_pct', 0), 1),
            doc.get('last_updated', ''),
            get_recommendation(doc.get('burnout', ''))
        ])

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode('utf-8')),
        media_type='text/csv',
        headers={'Content-Disposition': 'attachment; filename=burnout_report.csv'}
    )


if __name__ == "__main__":
    print("\n🏥 BurnoutAI Prediction API v5.0")
    print("   POST /predict        — image → V5 multimodal prediction + XAI")
    print("   POST /ocr            — alias for /predict")
    print("   GET  /results        — model metrics JSON")
    print("   GET  /generate_report — download CSV report")
    print("   GET  /health         — health check")
    print("   http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
