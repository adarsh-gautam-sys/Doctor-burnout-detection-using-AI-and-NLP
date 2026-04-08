"""
train_model_v4.py — BurnoutAI V4 Multimodal GPU Pipeline
=========================================================
Run:  python train_model_v4.py

Architecture (Fused from Handwritten_data.ipynb + burnout_v3):
  Phase 1: Frozen ViT/ResNet image embeddings  (GPU-accelerated)
  Phase 2: EasyOCR text extraction + NLP features
  Phase 3: Multimodal feature fusion
  Phase 4: XGBoost + MLP classifier with semi-supervised pseudo-labeling

Target: 70%+ OOF Accuracy on 129 labeled prescription images
Hardware: NVIDIA RTX 4060 (CUDA)
"""

import os, glob, warnings, random, json, datetime, sys, time
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path
from collections import Counter

from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

import joblib

warnings.filterwarnings('ignore')
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ══════════════════════════════════════════════════════════════
# PHASE 0 — GPU Setup & Paths
# ══════════════════════════════════════════════════════════════
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('=' * 60)
print('  BURNOUT AI — V4 MULTIMODAL GPU PIPELINE')
print('=' * 60)
print(f'  Python   : {sys.version.split()[0]}')
print(f'  PyTorch  : {torch.__version__}')
print(f'  Device   : {DEVICE.upper()}')
if DEVICE == 'cuda':
    print(f'  GPU      : {torch.cuda.get_device_name(0)}')
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'  VRAM     : {gpu_mem:.1f} GB')
print('=' * 60)

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
    print(f"  {'✓' if exists else '✗'} {name}: {p.name} {'✅' if exists else '❌'}")

labeled_imgs   = sorted(glob.glob(str(LABELED_DIR / '*.jpg')) + glob.glob(str(LABELED_DIR / '*.jpeg')) + glob.glob(str(LABELED_DIR / '*.png')))
unlabeled_imgs = sorted(glob.glob(str(UNLABELED_DIR / '*.jpg')) + glob.glob(str(UNLABELED_DIR / '*.jpeg')) + glob.glob(str(UNLABELED_DIR / '*.png')))
print(f'\n  ✓ Labeled images   : {len(labeled_imgs)}')
print(f'  ✓ Unlabeled images : {len(unlabeled_imgs)}')


# ── Load CSV ──────────────────────────────────────────────
df = pd.read_csv(CSV_PATH, encoding='utf-8-sig', header=None, names=['filename', 'label'])
df['filename'] = df['filename'].str.strip()
df['label'] = df['label'].str.strip()

print(f'\n  Class distribution:')
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
print(f'  ✓ Images found : {len(df) - missing}/{len(df)}')
df = df.dropna(subset=['full_path']).reset_index(drop=True)

le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])
print(f'  Class mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}')


# ══════════════════════════════════════════════════════════════
# PHASE 1 — Vision Embeddings (Frozen Pre-trained Model on GPU)
# ══════════════════════════════════════════════════════════════
print('\n' + '─' * 60)
print('  PHASE 1 — DEEP VISION EMBEDDINGS (GPU)')
print('─' * 60)

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print('  ⚠ timm not installed, using torchvision ResNet50')

IMGSIZE = 224

# ImageNet normalization
vision_transform = T.Compose([
    T.Resize((IMGSIZE, IMGSIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Augmented transforms for training
augmented_transform = T.Compose([
    T.Resize((IMGSIZE + 32, IMGSIZE + 32)),
    T.RandomCrop(IMGSIZE),
    T.RandomHorizontalFlip(p=0.1),
    T.RandomRotation(degrees=5),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    T.RandomAffine(degrees=3, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class PrescriptionDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMGSIZE, IMGSIZE), 128)
        if self.transform:
            img = self.transform(img)
        return img


# ── Build Backbone ────────────────────────────────────────
if HAS_TIMM:
    # Use EfficientNet-B3 (better than MobileNetV2, lighter than ViT)
    backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
    feat_dim = backbone.num_features
    print(f'  ✓ Backbone   : EfficientNet-B3 (timm)')
else:
    import torchvision.models as models
    _resnet = models.resnet50(weights='IMAGENET1K_V2')
    backbone = nn.Sequential(*list(_resnet.children())[:-1], nn.Flatten())
    feat_dim = 2048
    print(f'  ✓ Backbone   : ResNet-50 (torchvision)')

backbone = backbone.to(DEVICE).eval()
for p in backbone.parameters():
    p.requires_grad = False
print(f'  ✓ Feature dim: {feat_dim}')
print(f'  ✓ All weights FROZEN (no overfitting risk)')


def extract_vision_features(paths, desc='Extracting', use_augmentation=False, n_augments=5):
    """Extract deep vision features from a list of image paths.
    
    If use_augmentation=True, extract n_augments augmented versions 
    and average the embeddings for a more robust representation.
    """
    xform = vision_transform  # always use clean transform for extraction
    ds = PrescriptionDataset(paths, transform=xform)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    feats = []
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(DEVICE)
            out = backbone(batch)
            if out.dim() > 2:
                out = out.view(out.size(0), -1)
            feats.append(out.cpu().numpy())
    base_feats = np.vstack(feats)
    
    if not use_augmentation:
        print(f'  {desc}: {len(paths)} images → {base_feats.shape}')
        return base_feats
    
    # Test-Time Augmentation: average multiple augmented passes
    all_aug_feats = [base_feats]
    for aug_i in range(n_augments):
        ds_aug = PrescriptionDataset(paths, transform=augmented_transform)
        dl_aug = DataLoader(ds_aug, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
        aug_feats = []
        with torch.no_grad():
            for batch in dl_aug:
                batch = batch.to(DEVICE)
                out = backbone(batch)
                if out.dim() > 2:
                    out = out.view(out.size(0), -1)
                aug_feats.append(out.cpu().numpy())
        all_aug_feats.append(np.vstack(aug_feats))
    
    # Average all augmented embeddings
    averaged = np.mean(all_aug_feats, axis=0)
    print(f'  {desc}: {len(paths)} images × {n_augments + 1} augments → {averaged.shape}')
    return averaged


# ── Extract Labeled Vision Features ──────────────────────
print('\n  Extracting vision features from labeled images...')
t0 = time.time()
X_vision_labeled = extract_vision_features(
    df['full_path'].tolist(), desc='Labeled',
    use_augmentation=True, n_augments=5
)
print(f'  ⏱ Took {time.time()-t0:.1f}s')


# ══════════════════════════════════════════════════════════════
# PHASE 2 — OCR Text Extraction + NLP Features
# ══════════════════════════════════════════════════════════════
print('\n' + '─' * 60)
print('  PHASE 2 — OCR TEXT + NLP FEATURES')
print('─' * 60)

try:
    import easyocr
    HAS_OCR = True
    print('  ✓ EasyOCR loaded')
except ImportError:
    HAS_OCR = False
    print('  ⚠ EasyOCR not installed — skipping text features')

if HAS_OCR:
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), verbose=False)

    def extract_text_from_image(img_path):
        """Extract text using EasyOCR and return raw text + confidence."""
        try:
            results = reader.readtext(str(img_path), detail=1)
            texts = [r[1] for r in results]
            confidences = [r[2] for r in results]
            full_text = ' '.join(texts)
            avg_conf = np.mean(confidences) if confidences else 0.0
            return full_text, avg_conf, len(results)
        except Exception:
            return '', 0.0, 0

    def extract_nlp_features(text, ocr_confidence, num_detections):
        """Extract handcrafted NLP features from OCR text."""
        features = []
        
        # Basic text stats
        words = text.split()
        features.append(len(text))                           # total chars
        features.append(len(words))                          # word count
        features.append(np.mean([len(w) for w in words]) if words else 0)  # avg word length
        features.append(np.std([len(w) for w in words]) if len(words) > 1 else 0)  # word length std
        features.append(max([len(w) for w in words]) if words else 0)  # max word length
        features.append(min([len(w) for w in words]) if words else 0)  # min word length
        
        # Character distribution
        alpha_count = sum(c.isalpha() for c in text)
        digit_count = sum(c.isdigit() for c in text)
        space_count = sum(c.isspace() for c in text)
        special_count = len(text) - alpha_count - digit_count - space_count
        total = len(text) + 1e-8
        features.append(alpha_count / total)                 # alpha ratio
        features.append(digit_count / total)                 # digit ratio
        features.append(special_count / total)               # special char ratio
        features.append(space_count / total)                 # whitespace ratio
        
        # Capitalization
        upper_count = sum(c.isupper() for c in text)
        features.append(upper_count / (alpha_count + 1e-8))  # uppercase ratio
        
        # Abbreviation detection (short words <=3 chars)
        short_words = [w for w in words if len(w) <= 3]
        features.append(len(short_words) / (len(words) + 1e-8))  # abbreviation ratio
        
        # Medical-specific patterns
        # Count numbers that look like dosages (e.g., "500", "250mg")
        import re
        dosage_pattern = re.compile(r'\d+\s*(mg|ml|mcg|g|tab|cap|times|x)', re.IGNORECASE)
        dosage_count = len(dosage_pattern.findall(text))
        features.append(dosage_count)
        
        # Count potential drug-like words (mixed case, specific patterns)
        features.append(sum(1 for w in words if any(c.isupper() for c in w[1:]) if len(w) > 1) / (len(words) + 1e-8))
        
        # Sentence structure
        sentences = text.split('.')
        features.append(len([s for s in sentences if s.strip()]))  # sentence count
        features.append(np.mean([len(s.split()) for s in sentences if s.strip()]) if any(s.strip() for s in sentences) else 0)
        
        # OCR metadata
        features.append(ocr_confidence)                      # OCR confidence
        features.append(num_detections)                       # number of detected text regions
        
        # Lexical diversity
        unique_words = set(w.lower() for w in words)
        features.append(len(unique_words) / (len(words) + 1e-8))  # type-token ratio
        
        # Text density (chars per detection)
        features.append(len(text) / (num_detections + 1e-8))
        
        return np.array(features, dtype=np.float32)

    # Extract OCR + NLP features for labeled images
    print('  Extracting OCR text from labeled images...')
    nlp_features_list = []
    ocr_texts = []
    for i, path in enumerate(df['full_path'].tolist()):
        if (i+1) % 20 == 0 or i == 0:
            print(f'    OCR: {i+1}/{len(df)}', end='\r')
        text, conf, n_det = extract_text_from_image(path)
        ocr_texts.append(text)
        nlp_features_list.append(extract_nlp_features(text, conf, n_det))
    print(f'    OCR: {len(df)}/{len(df)} ✓         ')
    X_nlp_labeled = np.vstack(nlp_features_list)
    print(f'  ✓ NLP features shape: {X_nlp_labeled.shape}')
else:
    X_nlp_labeled = np.zeros((len(df), 1))
    print('  ⚠ Using dummy NLP features (no OCR)')


# ══════════════════════════════════════════════════════════════
# PHASE 2.5 — Handcrafted Image Features (from V3, but improved)
# ══════════════════════════════════════════════════════════════
print('\n' + '─' * 60)
print('  PHASE 2.5 — HANDCRAFTED IMAGE FEATURES')
print('─' * 60)

def extract_handcrafted_features(img_path):
    """Extract refined handcrafted features focusing on writing-quality signals."""
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception:
        return np.zeros(40)

    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    gray = np.array(img.convert('L'), dtype=np.float32) / 255.0
    h, w = gray.shape

    features = []

    # 1. Grayscale statistics (4)
    features.append(gray.mean())
    features.append(gray.std())
    mean_val = gray.mean()
    std_val = gray.std() + 1e-8
    features.append(float(((gray - mean_val) ** 3).mean() / (std_val ** 3)))  # skewness
    features.append(float(((gray - mean_val) ** 4).mean() / (std_val ** 4)))  # kurtosis

    # 2. Edge density (4) — indicates writing pressure/speed
    edges_x = np.abs(np.diff(gray, axis=1))
    edges_y = np.abs(np.diff(gray, axis=0))
    features.append(edges_x.mean())
    features.append(edges_x.std())
    features.append(edges_y.mean())
    features.append(edges_y.std())

    # 3. Ink density (8) — key burnout signal
    threshold = gray.mean()
    binary = (gray < threshold).astype(np.float32)
    features.append(binary.mean())        # overall ink ratio
    features.append(binary.sum())         # total ink pixels
    features.append(binary[:h//2, :w//2].mean())  # Q1 ink
    features.append(binary[:h//2, w//2:].mean())   # Q2 ink
    features.append(binary[h//2:, :w//2].mean())   # Q3 ink
    features.append(binary[h//2:, w//2:].mean())   # Q4 ink
    row_runs = np.diff(binary.astype(int), axis=1)
    features.append(np.abs(row_runs).sum() / (h * w))  # horizontal transition density
    col_runs = np.diff(binary.astype(int), axis=0)
    features.append(np.abs(col_runs).sum() / (h * w))  # vertical transition density

    # 4. Spatial uniformity (8) — rushed writing is less uniform
    row_means = gray.mean(axis=1)
    col_means = gray.mean(axis=0)
    features.append(row_means.std())
    features.append(col_means.std())
    features.append(np.diff(row_means).std())  # row smoothness
    features.append(np.diff(col_means).std())  # col smoothness
    
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

    # 5. Writing entropy proxy (4) — burnout correlates with entropy
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

    # 6. Image metadata (4)
    orig = Image.open(img_path)
    ow, oh = orig.size
    features.append(ow / (oh + 1e-8))
    features.append(float(ow * oh))
    features.append(float(ow))
    features.append(float(oh))

    # 7. Color channel stats (4) — ink color can vary
    for ch in range(3):
        features.append(arr[:,:,ch].mean() / 255.0)
    features.append(arr.std() / 255.0)

    return np.array(features, dtype=np.float32)


print('  Extracting handcrafted features...')
hc_features = []
for i, path in enumerate(df['full_path'].tolist()):
    if (i+1) % 50 == 0 or i == 0:
        print(f'    Handcrafted: {i+1}/{len(df)}', end='\r')
    hc_features.append(extract_handcrafted_features(path))
print(f'    Handcrafted: {len(df)}/{len(df)} ✓         ')
X_hc_labeled = np.vstack(hc_features)
print(f'  ✓ Handcrafted features shape: {X_hc_labeled.shape}')


# ══════════════════════════════════════════════════════════════
# PHASE 3 — MULTIMODAL FEATURE FUSION
# ══════════════════════════════════════════════════════════════
print('\n' + '─' * 60)
print('  PHASE 3 — MULTIMODAL FUSION')
print('─' * 60)

y_labeled = df['label_enc'].values

# Concatenate all feature modalities
X_labeled_raw = np.hstack([X_vision_labeled, X_nlp_labeled, X_hc_labeled])
print(f'  ✓ Vision features  : {X_vision_labeled.shape[1]}')
print(f'  ✓ NLP features     : {X_nlp_labeled.shape[1]}')
print(f'  ✓ Handcrafted      : {X_hc_labeled.shape[1]}')
print(f'  ✓ TOTAL fused (raw): {X_labeled_raw.shape[1]} dimensions')

# PCA to reduce dimensionality — prevents XGBoost OOM and reduces overfitting
N_COMPONENTS = min(128, X_labeled_raw.shape[0] - 1, X_labeled_raw.shape[1])
pca = PCA(n_components=N_COMPONENTS, random_state=SEED)
X_labeled = pca.fit_transform(X_labeled_raw)
print(f'  ✓ PCA reduced      : {X_labeled_raw.shape[1]} → {X_labeled.shape[1]} dims')
print(f'  ✓ Variance retained: {pca.explained_variance_ratio_.sum()*100:.1f}%')
print(f'  ✓ Labeled samples  : {X_labeled.shape[0]}')


# ══════════════════════════════════════════════════════════════
# PHASE 4 — BASELINE EVALUATION
# ══════════════════════════════════════════════════════════════
print('\n' + '─' * 60)
print('  PHASE 4 — BASELINE CLASSIFIERS')
print('─' * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# ── SVM Baseline ──────────────────────────────────────────
print('\n  Testing SVM...')
svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale',
                class_weight='balanced', probability=True, random_state=SEED))
])
svm_acc = cross_val_score(svm_pipe, X_labeled, y_labeled, cv=skf, scoring='accuracy', n_jobs=1)
svm_f1  = cross_val_score(svm_pipe, X_labeled, y_labeled, cv=skf, scoring='f1_macro', n_jobs=1)
print(f'  📊 SVM  — Acc: {svm_acc.mean()*100:.1f}% ± {svm_acc.std()*100:.1f}%  |  F1: {svm_f1.mean():.4f}')

# ── XGBoost ───────────────────────────────────────────────
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

if HAS_XGB:
    print('  Testing XGBoost...')
    xgb_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=2.0,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=SEED, n_jobs=1
        ))
    ])
    xgb_acc = cross_val_score(xgb_pipe, X_labeled, y_labeled, cv=skf, scoring='accuracy', n_jobs=1)
    xgb_f1  = cross_val_score(xgb_pipe, X_labeled, y_labeled, cv=skf, scoring='f1_macro', n_jobs=1)
    print(f'  📊 XGB  — Acc: {xgb_acc.mean()*100:.1f}% ± {xgb_acc.std()*100:.1f}%  |  F1: {xgb_f1.mean():.4f}')

# ── SVM with RBF + tuning ────────────────────────────────
print('  Testing SVM (tuned C=50, gamma=auto)...')
svm_tuned = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=50, gamma='auto',
                class_weight='balanced', probability=True, random_state=SEED))
])
svm2_acc = cross_val_score(svm_tuned, X_labeled, y_labeled, cv=skf, scoring='accuracy', n_jobs=1)
svm2_f1  = cross_val_score(svm_tuned, X_labeled, y_labeled, cv=skf, scoring='f1_macro', n_jobs=1)
print(f'  📊 SVM2 — Acc: {svm2_acc.mean()*100:.1f}% ± {svm2_acc.std()*100:.1f}%  |  F1: {svm2_f1.mean():.4f}')


# ══════════════════════════════════════════════════════════════
# PHASE 5 — SEMI-SUPERVISED PSEUDO-LABELING
# ══════════════════════════════════════════════════════════════
print('\n' + '─' * 60)
print('  PHASE 5 — SEMI-SUPERVISED PSEUDO-LABELING')
print('─' * 60)

print('  Extracting vision features from unlabeled images...')
t0 = time.time()
X_vision_unlabeled = extract_vision_features(unlabeled_imgs[:200], desc='Unlabeled (200 max)')
print(f'  ⏱ Took {time.time()-t0:.1f}s')

# Extract handcrafted features for unlabeled
unlabeled_subset = unlabeled_imgs[:200]  # cap to speed up
print('  Extracting handcrafted features from unlabeled images...')
hc_unlabeled = []
for i, path in enumerate(unlabeled_subset):
    if (i+1) % 100 == 0 or i == 0:
        print(f'    Handcrafted: {i+1}/{len(unlabeled_subset)}', end='\r')
    hc_unlabeled.append(extract_handcrafted_features(path))
print(f'    Handcrafted: {len(unlabeled_subset)}/{len(unlabeled_subset)} ✓         ')
X_hc_unlabeled = np.vstack(hc_unlabeled)

# Extract NLP features for unlabeled
if HAS_OCR:
    print('  Extracting OCR from unlabeled images...')
    nlp_unlabeled = []
    for i, path in enumerate(unlabeled_subset):
        if (i+1) % 50 == 0 or i == 0:
            print(f'    OCR: {i+1}/{len(unlabeled_subset)}', end='\r')
        text, conf, n_det = extract_text_from_image(path)
        nlp_unlabeled.append(extract_nlp_features(text, conf, n_det))
    print(f'    OCR: {len(unlabeled_subset)}/{len(unlabeled_subset)} ✓         ')
    X_nlp_unlabeled = np.vstack(nlp_unlabeled)
else:
    X_nlp_unlabeled = np.zeros((len(unlabeled_subset), X_nlp_labeled.shape[1]))

X_unlabeled_raw = np.hstack([X_vision_unlabeled, X_nlp_unlabeled, X_hc_unlabeled])
X_unlabeled = pca.transform(X_unlabeled_raw)  # use same PCA fit
print(f'  ✓ Unlabeled fused shape: {X_unlabeled.shape}')

# ── Multi-round Pseudo Labeling ──────────────────────────
CONFIDENCE_THRESHOLD = 0.80  # slightly lower to include more data

# Select best baseline model for pseudo labeling
all_scores = {'SVM': svm_f1.mean(), 'SVM_tuned': svm2_f1.mean()}
if HAS_XGB:
    all_scores['XGBoost'] = xgb_f1.mean()

best_baseline_name = max(all_scores, key=all_scores.get)
print(f'\n  ✓ Best baseline for pseudo-labeling: {best_baseline_name}')

# Build the pseudo-label teacher
if best_baseline_name == 'XGBoost':
    teacher = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=2.0,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=SEED, n_jobs=1
        ))
    ])
elif best_baseline_name == 'SVM_tuned':
    teacher = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=50, gamma='auto',
                    class_weight='balanced', probability=True, random_state=SEED))
    ])
else:
    teacher = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=10, gamma='scale',
                    class_weight='balanced', probability=True, random_state=SEED))
    ])

# Round 1: Pseudo label with original teacher
teacher.fit(X_labeled, y_labeled)
probs = teacher.predict_proba(X_unlabeled)
max_probs = probs.max(axis=1)
pseudo_preds = probs.argmax(axis=1)

confident_mask = max_probs >= CONFIDENCE_THRESHOLD
X_pseudo = X_unlabeled[confident_mask]
y_pseudo = pseudo_preds[confident_mask]

print(f'\n  📊 Pseudo Labeling (threshold={CONFIDENCE_THRESHOLD})')
print(f'    Total unlabeled        : {len(unlabeled_subset)}')
print(f'    High-confidence kept   : {confident_mask.sum()} ({confident_mask.mean()*100:.1f}%)')
for i, cls in enumerate(le.classes_):
    n = (y_pseudo == i).sum()
    print(f'      {cls:8s} : {n}')

# ── Combine ──────────────────────────────────────────────
X_combined = np.vstack([X_labeled, X_pseudo])
y_combined = np.concatenate([y_labeled, y_pseudo])
print(f'\n  ✓ Combined dataset : {len(X_combined)} images')


# ══════════════════════════════════════════════════════════════
# PHASE 6 — FINAL MODEL TRAINING + OOF EVALUATION
# ══════════════════════════════════════════════════════════════
print('\n' + '─' * 60)
print('  PHASE 6 — FINAL EVALUATION (OOF)')
print('─' * 60)

# Test all classifiers on combined data with OOF on labeled
classifiers = {}

# SVM
classifiers['SVM'] = lambda: Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=50, gamma='auto',
                class_weight='balanced', probability=True, random_state=SEED))
])

if HAS_XGB:
    classifiers['XGBoost'] = lambda: Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=3.0, min_child_weight=3,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=SEED, n_jobs=1
        ))
    ])

    classifiers['XGBoost_deep'] = lambda: Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', xgb.XGBClassifier(
            n_estimators=800, max_depth=5, learning_rate=0.01,
            subsample=0.7, colsample_bytree=0.6,
            reg_alpha=2.0, reg_lambda=5.0, min_child_weight=5,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=SEED, n_jobs=1
        ))
    ])

best_oof_acc = 0
best_oof_f1 = 0
best_name = ''
best_oof_preds = None

for clf_name, clf_factory in classifiers.items():
    print(f'\n  Evaluating {clf_name} (OOF on labeled, trained on combined)...')
    oof_preds = np.zeros(len(X_labeled), dtype=int)
    oof_probs = np.zeros((len(X_labeled), len(le.classes_)))
    
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_labeled, y_labeled)):
        clf = clf_factory()
        # Train on labeled-fold + pseudo labeled
        X_tr = np.vstack([X_labeled[tr_idx], X_pseudo])
        y_tr = np.concatenate([y_labeled[tr_idx], y_pseudo])
        clf.fit(X_tr, y_tr)
        oof_preds[val_idx] = clf.predict(X_labeled[val_idx])
        oof_probs[val_idx] = clf.predict_proba(X_labeled[val_idx])
    
    acc = accuracy_score(y_labeled, oof_preds)
    f1 = f1_score(y_labeled, oof_preds, average='macro')
    print(f'  📊 {clf_name:15s} — Acc: {acc*100:.2f}%  |  F1: {f1:.4f}')
    
    if f1 > best_oof_f1:
        best_oof_acc = acc
        best_oof_f1 = f1
        best_name = clf_name
        best_oof_preds = oof_preds.copy()

print(f'\n  ✅ BEST MODEL: {best_name}')
print(f'     OOF Accuracy : {best_oof_acc*100:.2f}%')
print(f'     OOF Macro F1 : {best_oof_f1:.4f}')

print('\n  ── Classification Report ──')
print(classification_report(y_labeled, best_oof_preds, target_names=le.classes_))

cm = confusion_matrix(y_labeled, best_oof_preds)


# ══════════════════════════════════════════════════════════════
# PHASE 7 — SAVE MODEL + RESULTS
# ══════════════════════════════════════════════════════════════
print('\n' + '─' * 60)
print('  PHASE 7 — SAVING MODEL')
print('─' * 60)

# Retrain best model on full combined data
final_model = classifiers[best_name]()
final_model.fit(X_combined, y_combined)

ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')
model_path   = SAVE_DIR / f'burnout_v4_{best_name}_{ts}.pkl'
encoder_path = SAVE_DIR / 'label_encoder_v4.pkl'

joblib.dump(final_model, model_path)
joblib.dump(le, encoder_path)
joblib.dump(final_model, SAVE_DIR / 'burnout_v4_latest.pkl')

print(f'  ✓ Model saved   : {model_path.name}')
print(f'  ✓ Latest        : burnout_v4_latest.pkl')
print(f'  ✓ Encoder       : label_encoder_v4.pkl')

# Save results JSON
report = classification_report(y_labeled, best_oof_preds, target_names=le.classes_, output_dict=True)

results = {
    "version": "4.0",
    "strategy": "Multimodal (Vision + NLP + Handcrafted) + Semi-Supervised",
    "backbone": "EfficientNet-B3 (frozen)" if HAS_TIMM else "ResNet-50 (frozen)",
    "ocr_engine": "EasyOCR" if HAS_OCR else "None",
    "classifier": best_name,
    "feature_dim": int(X_labeled.shape[1]),
    "feature_breakdown": {
        "vision": int(X_vision_labeled.shape[1]),
        "nlp": int(X_nlp_labeled.shape[1]),
        "handcrafted": int(X_hc_labeled.shape[1])
    },
    "labeled_images": int(len(X_labeled)),
    "pseudo_labeled": int(len(X_pseudo)),
    "total_training": int(len(X_combined)),
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "metrics": {
        "oof_accuracy": round(best_oof_acc * 100, 2),
        "oof_macro_f1": round(best_oof_f1, 4),
    },
    "per_class": {},
    "confusion_matrix": cm.tolist(),
    "version_comparison": {
        "V1_EfficientNet": {"accuracy": 44.2, "note": "CNN fine-tune, overfitting"},
        "V2_MobileNetV2": {"accuracy": 44.2, "note": "CNN fine-tune, same issue"},
        "V3_SemiSupervised": {"accuracy": 36.43, "note": "Handcrafted features + XGBoost"},
        "V4_Multimodal": {"accuracy": round(best_oof_acc * 100, 2), "note": f"{best_name} + Vision + NLP"}
    },
    "pseudo_label_distribution": {},
    "timestamp": ts,
    "gpu": torch.cuda.get_device_name(0) if DEVICE == 'cuda' else 'CPU'
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
print(f'  ✓ Results JSON : model_results.json')


# ══════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('  DOCTOR BURNOUT DETECTION — V4 RESULTS')
print('=' * 60)
print(f'  Strategy       : Multimodal (Vision + NLP + Handcrafted)')
print(f'  Vision Backbone: {"EfficientNet-B3" if HAS_TIMM else "ResNet-50"} (frozen, GPU)')
print(f'  OCR Engine     : {"EasyOCR" if HAS_OCR else "None"}')
print(f'  Classifier     : {best_name}')
print(f'  Feature Dim    : {X_labeled.shape[1]}')
print(f'  Labeled        : {len(X_labeled)}')
print(f'  Pseudo-labeled : {len(X_pseudo)}')
print(f'  Total training : {len(X_combined)}')
print(f'  GPU            : {torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"}')
print(f'')
print(f'  OOF Accuracy   : {best_oof_acc*100:.2f}%')
print(f'  OOF Macro F1   : {best_oof_f1:.4f}')
print(f'')
print(f'  vs V1 (CNN)    : {44.2}%')
print(f'  vs V3 (HC+XGB) : {36.43}%')
print(f'  IMPROVEMENT    : +{(best_oof_acc - 0.442)*100:.1f}% over V1')
print('=' * 60)
print('\n✅ Done!')
