"""
train_model.py — BurnoutAI V3 Semi-Supervised Training (No PyTorch)
=====================================================================
Run once:  python train_model.py

Same semi-supervised strategy as the notebook, but uses handcrafted
image features instead of MobileNetV2 (to avoid the 200MB PyTorch download).

Feature vector (per image, ~580 dims):
  - Color histograms (RGB, 3×64 = 192)
  - HSV histograms (3×64 = 192)
  - Grayscale stats (mean, std, skew, kurtosis = 4)
  - Edge density (Sobel-based = 4)
  - Texture features (LBP histogram = 10)
  - Spatial features (grid mean/std, 4×4×2 = 32)
  - Ink density features (binarization stats = 8)
  - Aspect ratio + size stats (4)

Pipeline:
  1. Extract features from all images
  2. SVM baseline on 129 labeled
  3. Pseudo-label ~384 unlabeled (confidence > 0.85)
  4. Retrain on combined
  5. Save best model + results JSON
"""

import os, glob, warnings, random, json, datetime, sys
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from pathlib import Path
from collections import Counter

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.pipeline import Pipeline

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠ xgboost not installed, will use SVM only")

import joblib

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print('✓ Running on CPU (no PyTorch needed)')
print(f'✓ Python: {sys.version.split()[0]}')


# ── Paths ─────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATASET_DIR   = BASE_DIR.parent / 'CliniCare Dataset'
LABELED_DIR   = DATASET_DIR / 'real'
UNLABELED_DIR = DATASET_DIR / 'dataset_prescrition'
CSV_PATH      = DATASET_DIR / 'real' / 'real_labels.csv'
SAVE_DIR      = BASE_DIR / 'data' / 'models'
SAVE_DIR.mkdir(parents=True, exist_ok=True)

for name, p in [('Labeled dir', LABELED_DIR), ('Unlabeled dir', UNLABELED_DIR), ('CSV', CSV_PATH)]:
    exists = p.exists()
    print(f"{'✓' if exists else '✗'} {name}: {p.name} {'✅' if exists else '❌'}")

labeled_imgs   = sorted(glob.glob(str(LABELED_DIR / '*.jpg')) + glob.glob(str(LABELED_DIR / '*.jpeg')) + glob.glob(str(LABELED_DIR / '*.png')))
unlabeled_imgs = sorted(glob.glob(str(UNLABELED_DIR / '*.jpg')) + glob.glob(str(UNLABELED_DIR / '*.jpeg')) + glob.glob(str(UNLABELED_DIR / '*.png')))
print(f'\n✓ Labeled images   : {len(labeled_imgs)}')
print(f'✓ Unlabeled images : {len(unlabeled_imgs)}')


# ── Load CSV ──────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, encoding='utf-8-sig', header=None, names=['filename', 'label'])
df['filename'] = df['filename'].str.strip()
df['label'] = df['label'].str.strip()

print(f'\nClass distribution:')
print(df['label'].value_counts().to_string())


def resolve_path(fname):
    fname = str(fname).strip()
    p = LABELED_DIR / fname
    if p.exists(): return str(p)
    for ext in ['.jpg', '.jpeg', '.png']:
        p2 = LABELED_DIR / (Path(fname).stem + ext)
        if p2.exists(): return str(p2)
    return None

df['full_path'] = df['filename'].apply(resolve_path)
missing = df['full_path'].isna().sum()
print(f'✓ Images found : {len(df) - missing}/{len(df)}')
df = df.dropna(subset=['full_path']).reset_index(drop=True)

le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])
print(f'Class mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}')


# ── Feature Extraction (No PyTorch) ──────────────────────
IMGSIZE = 224

def extract_image_features(img_path):
    """Extract a rich feature vector from a prescription image."""
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception:
        return np.zeros(588)  # fallback

    img = img.resize((IMGSIZE, IMGSIZE))
    arr = np.array(img, dtype=np.float32)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

    features = []

    # 1. Color histograms (3 × 64 = 192)
    for ch in [r, g, b]:
        hist, _ = np.histogram(ch, bins=64, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
        features.extend(hist)

    # 2. HSV histograms (3 × 64 = 192)
    hsv = img.convert('HSV')
    hsv_arr = np.array(hsv, dtype=np.float32)
    for ch_idx in range(3):
        ch = hsv_arr[:,:,ch_idx]
        hist, _ = np.histogram(ch, bins=64, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-8)
        features.extend(hist)

    # 3. Grayscale stats (4)
    gray = np.array(img.convert('L'), dtype=np.float32) / 255.0
    features.append(gray.mean())
    features.append(gray.std())
    # Skewness
    mean_val = gray.mean()
    std_val = gray.std() + 1e-8
    features.append(float(((gray - mean_val) ** 3).mean() / (std_val ** 3)))
    # Kurtosis
    features.append(float(((gray - mean_val) ** 4).mean() / (std_val ** 4)))

    # 4. Edge density (4)
    edges_x = np.abs(np.diff(gray, axis=1))
    edges_y = np.abs(np.diff(gray, axis=0))
    features.append(edges_x.mean())
    features.append(edges_x.std())
    features.append(edges_y.mean())
    features.append(edges_y.std())

    # 5. Texture: simplified LBP-like features (26)
    #    Compare each pixel to its neighbors
    gray_uint8 = (gray * 255).astype(np.uint8)
    # Compute local contrast in 8x8 blocks
    h, w = gray.shape
    block_h, block_w = h // 8, w // 8
    block_means = []
    block_stds = []
    for bi in range(8):
        for bj in range(8):
            block = gray[bi*block_h:(bi+1)*block_h, bj*block_w:(bj+1)*block_w]
            block_means.append(block.mean())
            block_stds.append(block.std())
    # Keep statistical summary of blocks
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
    # Spatial variance across rows and columns
    row_means = gray.mean(axis=1)
    col_means = gray.mean(axis=0)
    features.append(row_means.std())
    features.append(col_means.std())
    # Row/col entropy (smoothness of left-right vs top-bottom writing)
    features.append(np.diff(row_means).std())
    features.append(np.diff(col_means).std())
    # Corner vs center intensity ratios
    center = gray[h//4:3*h//4, w//4:3*w//4].mean()
    corners_val = np.mean([gray[:h//4,:w//4].mean(), gray[:h//4,-w//4:].mean(),
                          gray[-h//4:,:w//4].mean(), gray[-h//4:,-w//4:].mean()])
    features.append(center / (corners_val + 1e-8))
    features.append(center - corners_val)

    # Quadrant differences (writing uniformity)
    q1 = gray[:h//2, :w//2].mean()
    q2 = gray[:h//2, w//2:].mean()
    q3 = gray[h//2:, :w//2].mean()
    q4 = gray[h//2:, w//2:].mean()
    features.extend([q1, q2, q3, q4])
    features.append(max(q1,q2,q3,q4) - min(q1,q2,q3,q4))  # max diff
    features.append(np.std([q1,q2,q3,q4]))

    # Horizontal vs vertical profile
    features.append(row_means.max() - row_means.min())
    features.append(col_means.max() - col_means.min())

    # 6. Ink density features (8)
    #    Binarize (Otsu-like threshold)
    threshold = gray.mean()
    binary = (gray < threshold).astype(np.float32)  # ink is dark
    features.append(binary.mean())  # ink ratio
    features.append(binary.sum())   # total ink pixels

    # Ink coverage per quadrant
    features.append(binary[:h//2, :w//2].mean())
    features.append(binary[:h//2, w//2:].mean())
    features.append(binary[h//2:, :w//2].mean())
    features.append(binary[h//2:, w//2:].mean())

    # Connected component proxy (run-length)
    row_runs = np.diff(binary.astype(int), axis=1)
    features.append(np.abs(row_runs).sum() / (h * w))  # transition density
    col_runs = np.diff(binary.astype(int), axis=0)
    features.append(np.abs(col_runs).sum() / (h * w))

    # 7. Size/aspect (4)
    orig = Image.open(img_path)
    ow, oh = orig.size
    features.append(ow / (oh + 1e-8))  # aspect ratio
    features.append(float(ow * oh))     # total pixels
    features.append(float(ow))
    features.append(float(oh))

    # 8. Frequency domain proxy — DCT-like via patch variances (64)
    #    4×4 grid of 4×4 subblocks, each block's variance in a row
    patch_size = IMGSIZE // 8
    freq_features = []
    for pi in range(8):
        for pj in range(8):
            patch = gray[pi*patch_size:(pi+1)*patch_size, pj*patch_size:(pj+1)*patch_size]
            freq_features.append(patch.var())
    features.extend(freq_features)

    return np.array(features, dtype=np.float32)


def extract_features_batch(paths, desc='Extracting'):
    """Extract features from a list of image paths."""
    features = []
    total = len(paths)
    for i, p in enumerate(paths):
        if (i + 1) % 50 == 0 or i == 0:
            print(f'  {desc}: {i+1}/{total}', end='\r')
        features.append(extract_image_features(p))
    print(f'  {desc}: {total}/{total} ✓    ')
    return np.vstack(features)


# ── Extract Labeled Features ─────────────────────────────
print('\nExtracting features from labeled images...')
X_labeled = extract_features_batch(df['full_path'].tolist(), desc='Labeled')
y_labeled = df['label_enc'].values
print(f'✓ Labeled features shape : {X_labeled.shape}')


# ── Baseline SVM ─────────────────────────────────────────
print('\nTraining baseline SVM...')

svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    SVC(kernel='rbf', C=10, gamma='scale',
                   class_weight='balanced', probability=True, random_state=SEED))
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

baseline_scores = cross_val_score(svm_pipe, X_labeled, y_labeled,
                                   cv=skf, scoring='f1_macro', n_jobs=-1)
baseline_acc    = cross_val_score(svm_pipe, X_labeled, y_labeled,
                                   cv=skf, scoring='accuracy', n_jobs=-1)

print(f'\n📊 BASELINE SVM (labeled only)')
print(f'   Accuracy  : {baseline_acc.mean()*100:.1f}% ± {baseline_acc.std()*100:.1f}%')
print(f'   Macro F1  : {baseline_scores.mean():.4f} ± {baseline_scores.std():.4f}')


# ── Extract Unlabeled Features ───────────────────────────
print(f'\nExtracting features from {len(unlabeled_imgs)} unlabeled images...')
X_unlabeled = extract_features_batch(unlabeled_imgs, desc='Unlabeled')
print(f'✓ Unlabeled features shape : {X_unlabeled.shape}')


# ── Pseudo Labeling ──────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.85

svm_pipe.fit(X_labeled, y_labeled)
probs        = svm_pipe.predict_proba(X_unlabeled)
max_probs    = probs.max(axis=1)
pseudo_preds = probs.argmax(axis=1)

confident_mask = max_probs >= CONFIDENCE_THRESHOLD
X_pseudo       = X_unlabeled[confident_mask]
y_pseudo       = pseudo_preds[confident_mask]

print(f'\n📊 Pseudo Labeling (threshold={CONFIDENCE_THRESHOLD})')
print(f'   Total unlabeled      : {len(unlabeled_imgs)}')
print(f'   High-confidence kept : {confident_mask.sum()} ({confident_mask.mean()*100:.1f}%)')
print(f'   Discarded            : {(~confident_mask).sum()}')
for i, cls in enumerate(le.classes_):
    n = (y_pseudo == i).sum()
    print(f'     {cls:8s} : {n}')


# ── Combine ──────────────────────────────────────────────
X_combined = np.vstack([X_labeled, X_pseudo])
y_combined = np.concatenate([y_labeled, y_pseudo])

print(f'\n✓ Combined dataset : {len(X_combined)} images')


# ── Train SVM on Combined ────────────────────────────────
print('\nTraining SVM on combined data...')

svm_combined = Pipeline([
    ('scaler', StandardScaler()),
    ('svm',    SVC(kernel='rbf', C=10, gamma='scale',
                   class_weight='balanced', probability=True, random_state=SEED))
])

combined_acc = cross_val_score(svm_combined, X_labeled, y_labeled, cv=skf, scoring='accuracy', n_jobs=-1)
combined_f1  = cross_val_score(svm_combined, X_labeled, y_labeled, cv=skf, scoring='f1_macro', n_jobs=-1)
svm_combined.fit(X_combined, y_combined)

print(f'\n📊 SEMI-SUPERVISED SVM')
print(f'   Accuracy  : {combined_acc.mean()*100:.1f}% ± {combined_acc.std()*100:.1f}%')
print(f'   Macro F1  : {combined_f1.mean():.4f} ± {combined_f1.std():.4f}')


# ── XGBoost ──────────────────────────────────────────────
best_f1_score = combined_f1.mean()
best_model = svm_combined
best_name = 'SVM'
xgb_acc_val = None
xgb_f1_val = None

if HAS_XGB:
    print('\nAlso testing XGBoost...')
    xgb_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb',    xgb.XGBClassifier(n_estimators=200, max_depth=4,
                                       learning_rate=0.05, subsample=0.8,
                                       use_label_encoder=False,
                                       eval_metric='mlogloss',
                                       random_state=SEED, n_jobs=-1))
    ])
    xgb_acc_cv = cross_val_score(xgb_pipe, X_labeled, y_labeled, cv=skf, scoring='accuracy', n_jobs=-1)
    xgb_f1_cv  = cross_val_score(xgb_pipe, X_labeled, y_labeled, cv=skf, scoring='f1_macro', n_jobs=-1)
    xgb_acc_val = xgb_acc_cv.mean()
    xgb_f1_val = xgb_f1_cv.mean()

    print(f'\n📊 XGBoost')
    print(f'   Accuracy  : {xgb_acc_val*100:.1f}% ± {xgb_acc_cv.std()*100:.1f}%')
    print(f'   Macro F1  : {xgb_f1_val:.4f} ± {xgb_f1_cv.std():.4f}')

    if xgb_f1_val > best_f1_score:
        xgb_pipe.fit(X_combined, y_combined)
        best_model = xgb_pipe
        best_name = 'XGBoost'
        best_f1_score = xgb_f1_val

print(f'\n✓ Best model: {best_name}')


# ── OOF Evaluation ───────────────────────────────────────
oof_preds = np.zeros(len(X_labeled), dtype=int)
oof_probs = np.zeros((len(X_labeled), 3))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_labeled, y_labeled)):
    if best_name == 'SVM':
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svm',    SVC(kernel='rbf', C=10, gamma='scale',
                           class_weight='balanced', probability=True, random_state=SEED))
        ])
    else:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb',    xgb.XGBClassifier(n_estimators=200, max_depth=4,
                                           learning_rate=0.05, subsample=0.8,
                                           use_label_encoder=False,
                                           eval_metric='mlogloss',
                                           random_state=SEED, n_jobs=-1))
        ])
    X_tr = np.vstack([X_labeled[tr_idx], X_pseudo])
    y_tr = np.concatenate([y_labeled[tr_idx], y_pseudo])
    pipe.fit(X_tr, y_tr)
    oof_preds[val_idx] = pipe.predict(X_labeled[val_idx])
    oof_probs[val_idx] = pipe.predict_proba(X_labeled[val_idx])
    print(f'  Fold {fold+1} done')

oof_acc = accuracy_score(y_labeled, oof_preds)
oof_f1  = f1_score(y_labeled, oof_preds, average='macro')
per_class_f1 = f1_score(y_labeled, oof_preds, average=None)
cm = confusion_matrix(y_labeled, oof_preds)

print(f'\n✓ OOF Accuracy : {oof_acc*100:.2f}%')
print(f'✓ OOF Macro F1 : {oof_f1:.4f}')

print('\n── Classification Report ──')
print(classification_report(y_labeled, oof_preds, target_names=le.classes_))


# ── Save Model ───────────────────────────────────────────
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')
model_path   = SAVE_DIR / f'burnout_v3_{best_name}_{ts}.pkl'
encoder_path = SAVE_DIR / 'label_encoder_v3.pkl'

joblib.dump(best_model, model_path)
joblib.dump(le, encoder_path)
joblib.dump(best_model, SAVE_DIR / 'burnout_v3_latest.pkl')

print(f'\n✓ Model saved : {model_path.name}')
print(f'✓ Latest      : burnout_v3_latest.pkl')
print(f'✓ Encoder     : label_encoder_v3.pkl')


# ── Save Results JSON ────────────────────────────────────
report = classification_report(y_labeled, oof_preds, target_names=le.classes_, output_dict=True)

results = {
    "version": "3.0",
    "strategy": "Semi-Supervised Learning",
    "backbone": "Handcrafted Features (color + texture + spatial + ink density)",
    "classifier": best_name,
    "feature_dim": int(X_labeled.shape[1]),
    "labeled_images": int(len(X_labeled)),
    "pseudo_labeled": int(len(X_pseudo)),
    "total_training": int(len(X_combined)),
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "metrics": {
        "oof_accuracy": round(oof_acc * 100, 2),
        "oof_macro_f1": round(oof_f1, 4),
        "baseline_accuracy": round(baseline_acc.mean() * 100, 2),
        "baseline_f1": round(baseline_scores.mean(), 4),
        "improvement_over_v1v2": round((oof_acc - 0.442) * 100, 1)
    },
    "per_class": {},
    "confusion_matrix": cm.tolist(),
    "version_comparison": {
        "V1_EfficientNet": {"accuracy": 44.2, "note": "CNN fine-tune, overfitting"},
        "V2_MobileNetV2": {"accuracy": 44.2, "note": "CNN fine-tune, same issue"},
        "V3_SemiSupervised": {"accuracy": round(oof_acc * 100, 2), "note": f"{best_name} + pseudo labels"}
    },
    "pseudo_label_distribution": {},
    "timestamp": ts
}

for cls in le.classes_:
    results["per_class"][cls] = {
        "precision": round(report[cls]["precision"], 4),
        "recall": round(report[cls]["recall"], 4),
        "f1": round(report[cls]["f1-score"], 4),
        "support": int(report[cls]["support"])
    }

for i, cls in enumerate(le.classes_):
    results["pseudo_label_distribution"][cls] = int((y_pseudo == i).sum())

results_path = BASE_DIR / 'data' / 'model_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f'✓ Results JSON : model_results.json')


# ── Final Summary ────────────────────────────────────────
print('\n' + '=' * 55)
print('   DOCTOR BURNOUT DETECTION — V3 RESULTS')
print('=' * 55)
print(f'   Strategy       : Semi-Supervised')
print(f'   Features       : Handcrafted ({X_labeled.shape[1]}-dim)')
print(f'   Classifier     : {best_name}')
print(f'   Labeled        : {len(X_labeled)}')
print(f'   Pseudo-labeled : {len(X_pseudo)}')
print(f'   Total training : {len(X_combined)}')
print(f'   OOF Accuracy   : {oof_acc*100:.2f}%')
print(f'   OOF Macro F1   : {oof_f1:.4f}')
print(f'   vs V1/V2       : +{(oof_acc - 0.442)*100:.1f}%')
print('=' * 55)
print('\n✅ Done!')
