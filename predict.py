"""
predict.py — BurnoutAI FastAPI Inference Server (No PyTorch)
==============================================================
Run:  python predict.py
Endpoints:
  POST /predict  — upload image → burnout prediction + XAI explanation
  POST /ocr      — alias for /predict (frontend compatibility)
  GET  /results  — returns model_results.json
  GET  /generate_report — download CSV report of all doctors
  GET  /health   — health check
"""

import os, json, csv, io
import numpy as np
from pathlib import Path
from io import BytesIO

from PIL import Image
import joblib

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import uvicorn

# ── Paths ─────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
MODELS_DIR   = BASE_DIR / 'data' / 'models'
RESULTS_PATH = BASE_DIR / 'data' / 'model_results.json'
DASHBOARD_PATH = BASE_DIR / 'data' / 'dashboard_data.json'


# ═══════════════════════════════════════════════════════════
# FEATURE NAME MAPPING (490-dim, matches train_model.py)
# ═══════════════════════════════════════════════════════════

def build_feature_names():
    """Build the 490 feature names matching extract_image_features() order."""
    names = []

    # 1. Color histograms: R(64) + G(64) + B(64) = 192
    for color in ['red', 'green', 'blue']:
        for i in range(64):
            names.append(f'{color}_hist_{i}')

    # 2. HSV histograms: H(64) + S(64) + V(64) = 192
    for channel in ['hue', 'saturation', 'value']:
        for i in range(64):
            names.append(f'{channel}_hist_{i}')

    # 3. Grayscale stats (4)
    names.extend(['gray_mean', 'gray_std', 'gray_skewness', 'gray_kurtosis'])

    # 4. Edge density (4)
    names.extend(['edge_x_mean', 'edge_x_std', 'edge_y_mean', 'edge_y_std'])

    # 5. Texture features (26)
    names.extend([
        'texture_block_mean', 'texture_block_std',
        'texture_contrast_mean', 'texture_contrast_std',
        'texture_block_p25', 'texture_block_p75',
        'texture_contrast_p25', 'texture_contrast_p75',
        'writing_row_variance', 'writing_col_variance',
        'writing_row_smoothness', 'writing_col_smoothness',
        'center_corner_ratio', 'center_corner_diff',
        'quadrant_1', 'quadrant_2', 'quadrant_3', 'quadrant_4',
        'writing_uniformity', 'writing_spread',
        'row_profile_range', 'col_profile_range'
    ])

    # 6. Ink density features (8)
    names.extend([
        'ink_density', 'ink_pixel_count',
        'ink_q1', 'ink_q2', 'ink_q3', 'ink_q4',
        'stroke_transition_h', 'stroke_transition_v'
    ])

    # 7. Size/aspect (4)
    names.extend(['aspect_ratio', 'total_pixels', 'image_width', 'image_height'])

    # 8. Frequency domain proxy (64)
    for i in range(64):
        names.append(f'freq_patch_{i}')

    return names

FEATURE_NAMES = build_feature_names()

# Human-readable display names for important features
FEATURE_DISPLAY = {
    'gray_mean': 'Grayscale Brightness',
    'gray_std': 'Contrast Variation',
    'gray_skewness': 'Tonal Asymmetry',
    'gray_kurtosis': 'Contrast Sharpness',
    'edge_x_mean': 'Horizontal Edge Density',
    'edge_x_std': 'Horizontal Stroke Irregularity',
    'edge_y_mean': 'Vertical Edge Density',
    'edge_y_std': 'Vertical Stroke Irregularity',
    'texture_block_mean': 'Overall Texture Density',
    'texture_block_std': 'Texture Variation',
    'texture_contrast_mean': 'Writing Pressure Average',
    'texture_contrast_std': 'Writing Pressure Inconsistency',
    'writing_row_variance': 'Line-to-Line Variation',
    'writing_col_variance': 'Column Spacing Variation',
    'writing_row_smoothness': 'Line Steadiness',
    'writing_col_smoothness': 'Character Spacing Steadiness',
    'center_corner_ratio': 'Writing Centrality',
    'center_corner_diff': 'Page Utilization Balance',
    'writing_uniformity': 'Quadrant Uniformity',
    'writing_spread': 'Writing Spatial Spread',
    'row_profile_range': 'Vertical Writing Density Range',
    'col_profile_range': 'Horizontal Writing Density Range',
    'ink_density': 'Ink Coverage Ratio',
    'ink_pixel_count': 'Total Ink Pixels',
    'ink_q1': 'Ink Density (Top-Left)',
    'ink_q2': 'Ink Density (Top-Right)',
    'ink_q3': 'Ink Density (Bottom-Left)',
    'ink_q4': 'Ink Density (Bottom-Right)',
    'stroke_transition_h': 'Horizontal Stroke Fragmentation',
    'stroke_transition_v': 'Vertical Stroke Fragmentation',
    'aspect_ratio': 'Document Aspect Ratio',
}


def get_feature_importance(model_pipeline):
    """Extract feature importance from the trained model pipeline.

    For XGBoost: uses model.feature_importances_ directly.
    For SVM: uses absolute value of coefficients averaged across classes.
    Returns array of shape (n_features,).
    """
    model = model_pipeline.named_steps.get('xgb') or model_pipeline.named_steps.get('svm')

    if hasattr(model, 'feature_importances_'):
        # XGBoost
        return model.feature_importances_
    elif hasattr(model, 'coef_'):
        # SVM with linear-like access
        return np.abs(model.coef_).mean(axis=0)
    else:
        # Fallback — uniform
        n = len(FEATURE_NAMES)
        return np.ones(n) / n


def get_top_features(feature_vector, importances, top_k=5):
    """Get the top-k most important features with their values and importance scores.

    Args:
        feature_vector: 1D array of extracted feature values (490,)
        importances: 1D array of feature importance scores (490,)
        top_k: number of top features to return

    Returns:
        List of dicts with feature_name, display_name, value, importance
    """
    # Filter to only interpretable features (skip histogram/freq bins)
    interpretable_indices = []
    for i, name in enumerate(FEATURE_NAMES):
        if not any(name.startswith(p) for p in ['red_hist', 'green_hist', 'blue_hist',
                                                  'hue_hist', 'saturation_hist', 'value_hist',
                                                  'freq_patch']):
            interpretable_indices.append(i)

    # Get importances for interpretable features only
    interp_importances = importances[interpretable_indices]
    interp_sorted = np.argsort(interp_importances)[::-1][:top_k]

    top = []
    max_imp = interp_importances.max() if interp_importances.max() > 0 else 1.0
    for rank_idx in interp_sorted:
        feat_idx = interpretable_indices[rank_idx]
        name = FEATURE_NAMES[feat_idx]
        display = FEATURE_DISPLAY.get(name, name.replace('_', ' ').title())
        top.append({
            'feature_name': name,
            'display_name': display,
            'value': round(float(feature_vector[feat_idx]), 4),
            'importance': round(float(importances[feat_idx] / max_imp), 4),
            'raw_importance': round(float(importances[feat_idx]), 6)
        })

    return top


def generate_explanation_summary(prediction, top_features, feature_vector):
    """Generate a human-readable summary using actual feature values.

    Rules are based on the real feature semantics from train_model.py.
    """
    findings = []
    fv = {FEATURE_NAMES[i]: float(feature_vector[i]) for i in range(len(feature_vector))}

    # Ink density analysis
    ink = fv.get('ink_density', 0)
    if ink > 0.55:
        findings.append('Heavy ink pressure detected — potential sign of motor tension')
    elif ink < 0.35:
        findings.append('Light ink coverage — possible fatigue or rushed writing')

    # Writing uniformity
    uniformity = fv.get('writing_uniformity', 0)
    if uniformity > 0.15:
        findings.append('Uneven writing distribution across page quadrants')

    # Stroke transitions (fragmentation)
    st_h = fv.get('stroke_transition_h', 0)
    st_v = fv.get('stroke_transition_v', 0)
    if st_h > 0.15 or st_v > 0.15:
        findings.append('Fragmented stroke patterns detected — irregular pen lifts')

    # Edge irregularity
    ex_std = fv.get('edge_x_std', 0)
    ey_std = fv.get('edge_y_std', 0)
    if ex_std > 0.05 or ey_std > 0.05:
        findings.append('Irregular stroke edges — potential hand tremor or instability')

    # Writing pressure inconsistency
    tc_std = fv.get('texture_contrast_std', 0)
    if tc_std > 0.03:
        findings.append('Inconsistent writing pressure across the document')

    # Row/column smoothness
    row_sm = fv.get('writing_row_smoothness', 0)
    col_sm = fv.get('writing_col_smoothness', 0)
    if row_sm > 0.02 or col_sm > 0.02:
        findings.append('Unsteady line progression — writing drifts between lines')

    # Grayscale skewness (unusual contrast)
    skew = fv.get('gray_skewness', 0)
    if abs(skew) > 1.5:
        findings.append('Unusual tonal distribution in handwriting contrast')

    # Center-corner balance
    cc_diff = fv.get('center_corner_diff', 0)
    if abs(cc_diff) > 0.1:
        findings.append('Writing concentrates unevenly on the page')

    # Add top feature mentions if not already covered
    for feat in top_features[:3]:
        display = feat['display_name']
        mention = f"High '{display}' signal contributed to prediction"
        if not any(display.lower() in f.lower() for f in findings):
            findings.append(mention)

    if not findings:
        findings.append('No strong individual indicators — prediction based on combined feature patterns')

    # Build final summary
    risk_text = {
        'High': 'High burnout risk detected',
        'Medium': 'Moderate burnout indicators present',
        'Low': 'Low burnout risk — writing appears stable'
    }
    summary_header = risk_text.get(prediction, f'{prediction} burnout level detected')

    return {
        'summary': f"{summary_header}. {findings[0]}.",
        'findings': findings[:5],
        'feature_count': len(feature_vector),
        'interpretable_features_analyzed': 106  # 490 - 384 histogram/freq bins
    }


# ── Feature Extraction (must match train_model.py exactly) ──
IMGSIZE = 224

def extract_image_features(img):
    """Extract feature vector from a PIL Image. Must match train_model.py."""
    img_resized = img.resize((IMGSIZE, IMGSIZE))
    arr = np.array(img_resized, dtype=np.float32)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    features = []

    # 1. Color histograms (3 × 64 = 192)
    for ch in [r, g, b]:
        hist, _ = np.histogram(ch, bins=64, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
        features.extend(hist)

    # 2. HSV histograms (3 × 64 = 192)
    hsv = img_resized.convert('HSV')
    hsv_arr = np.array(hsv, dtype=np.float32)
    for ch_idx in range(3):
        ch = hsv_arr[:,:,ch_idx]
        hist, _ = np.histogram(ch, bins=64, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
        features.extend(hist)

    # 3. Grayscale stats (4)
    gray = np.array(img_resized.convert('L'), dtype=np.float32) / 255.0
    features.append(gray.mean())
    features.append(gray.std())
    mean_val = gray.mean()
    std_val = gray.std() + 1e-8
    features.append(float(((gray - mean_val) ** 3).mean() / (std_val ** 3)))
    features.append(float(((gray - mean_val) ** 4).mean() / (std_val ** 4)))

    # 4. Edge density (4)
    edges_x = np.abs(np.diff(gray, axis=1))
    edges_y = np.abs(np.diff(gray, axis=0))
    features.append(edges_x.mean())
    features.append(edges_x.std())
    features.append(edges_y.mean())
    features.append(edges_y.std())

    # 5. Texture features (26)
    h, w = gray.shape
    block_h, block_w = h // 8, w // 8
    block_means = []
    block_stds = []
    for bi in range(8):
        for bj in range(8):
            block = gray[bi*block_h:(bi+1)*block_h, bj*block_w:(bj+1)*block_w]
            block_means.append(block.mean())
            block_stds.append(block.std())
    block_means = np.array(block_means)
    block_stds = np.array(block_stds)
    features.append(block_means.mean())
    features.append(block_means.std())
    features.append(block_stds.mean())
    features.append(block_stds.std())
    features.append(np.percentile(block_means, 25))
    features.append(np.percentile(block_means, 75))
    features.append(np.percentile(block_stds, 25))
    features.append(np.percentile(block_stds, 75))
    row_means = gray.mean(axis=1)
    col_means = gray.mean(axis=0)
    features.append(row_means.std())
    features.append(col_means.std())
    features.append(np.diff(row_means).std())
    features.append(np.diff(col_means).std())
    center = gray[h//4:3*h//4, w//4:3*w//4].mean()
    corners_val = np.mean([gray[:h//4,:w//4].mean(), gray[:h//4,-w//4:].mean(),
                          gray[-h//4:,:w//4].mean(), gray[-h//4:,-w//4:].mean()])
    features.append(center / (corners_val + 1e-8))
    features.append(center - corners_val)
    q1 = gray[:h//2, :w//2].mean()
    q2 = gray[:h//2, w//2:].mean()
    q3 = gray[h//2:, :w//2].mean()
    q4 = gray[h//2:, w//2:].mean()
    features.extend([q1, q2, q3, q4])
    features.append(max(q1,q2,q3,q4) - min(q1,q2,q3,q4))
    features.append(np.std([q1,q2,q3,q4]))
    features.append(row_means.max() - row_means.min())
    features.append(col_means.max() - col_means.min())

    # 6. Ink density features (8)
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

    # 7. Size/aspect (4) — use resized size since we don't have original
    features.append(1.0)  # aspect ratio placeholder
    features.append(float(IMGSIZE * IMGSIZE))
    features.append(float(IMGSIZE))
    features.append(float(IMGSIZE))

    # 8. Frequency domain proxy (64)
    patch_size = IMGSIZE // 8
    for pi in range(8):
        for pj in range(8):
            patch = gray[pi*patch_size:(pi+1)*patch_size, pj*patch_size:(pj+1)*patch_size]
            features.append(patch.var())

    return np.array(features, dtype=np.float32)


# ── App ───────────────────────────────────────────────────
app = FastAPI(title="BurnoutAI Prediction API", version="3.1")

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

# Serve root files (index.html etc.)
app.mount("/static", StaticFiles(directory=str(BASE_DIR), html=True), name="static")

# Global state
classifier = None
label_encoder = None
feature_importances = None  # cached after model load


def load_models():
    global classifier, label_encoder, feature_importances

    model_path = MODELS_DIR / 'burnout_v3_latest.pkl'
    if not model_path.exists():
        v3_models = list(MODELS_DIR.glob('burnout_v3_*.pkl'))
        if v3_models:
            model_path = v3_models[0]
        else:
            print("⚠ No model found — run train_model.py first!")
            return

    print(f"Loading classifier: {model_path.name}")
    classifier = joblib.load(model_path)
    print("  ✓ Classifier loaded")

    # Cache feature importances at startup
    feature_importances = get_feature_importance(classifier)
    print(f"  ✓ Feature importances extracted ({len(feature_importances)} features)")

    encoder_path = MODELS_DIR / 'label_encoder_v3.pkl'
    if not encoder_path.exists():
        encoder_path = MODELS_DIR / 'label_encoder.pkl'
    if encoder_path.exists():
        label_encoder = joblib.load(encoder_path)
        print(f"  ✓ Label encoder: {list(label_encoder.classes_)}")
    else:
        print("  ⚠ No label encoder found")


@app.on_event("startup")
async def startup():
    load_models()


@app.get("/")
async def serve_index():
    return FileResponse(BASE_DIR / "index.html")

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": classifier is not None}


@app.get("/results")
async def get_results():
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return {"error": "No results file found"}


async def _predict_image(file: UploadFile):
    """Core prediction logic for image files — used by /predict and /ocr."""
    if classifier is None:
        raise HTTPException(503, "Model not loaded. Run train_model.py first.")

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

    fv = extract_image_features(img)
    features = fv.reshape(1, -1)

    proba = classifier.predict_proba(features)[0]
    pred_idx = proba.argmax()
    pred_label = label_encoder.classes_[pred_idx]

    probs = {}
    for i, cls in enumerate(label_encoder.classes_):
        probs[cls] = round(float(proba[i]), 4)

    # ── XAI Explanation ───────────────────────────────────
    top_feats = get_top_features(fv, feature_importances, top_k=5)
    explanation = generate_explanation_summary(pred_label, top_feats, fv)
    explanation['top_features'] = top_feats
    explanation['model_type'] = 'XGBoost' if hasattr(
        classifier.named_steps.get('xgb', object()), 'feature_importances_'
    ) else 'SVM'

    return {
        "prediction": pred_label,
        "burnout_risk": pred_label,
        "confidence": round(float(proba.max()) * 100, 1),
        "probabilities": probs,
        "input_type": "image",
        "filename": file.filename,
        "explanation": explanation
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

    # Recommendation logic
    def get_recommendation(risk_level):
        return {
            'High': 'Immediate intervention required. Reduce workload and arrange support.',
            'Medium': 'Monitor closely. Schedule follow-up assessment within 2 weeks.',
            'Low': 'No immediate action needed. Continue routine monitoring.'
        }.get(risk_level, 'Assessment pending')

    # Build CSV in memory
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
    print("\n🏥 BurnoutAI Prediction API v3.1")
    print("   POST /predict        — image → prediction + XAI")
    print("   POST /ocr            — alias for /predict")
    print("   GET  /results        — model metrics JSON")
    print("   GET  /generate_report — download CSV report")
    print("   GET  /health         — health check")
    print("   http://127.0.0.1:8000\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
