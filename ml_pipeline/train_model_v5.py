"""
BURNOUT AI — V5 MULTIMODAL GPU PIPELINE
========================================
  Phase 1: End-to-End Vision Fine-Tuning (partially unfrozen EfficientNet-B3)
  Phase 2: EasyOCR text extraction + NLP features
  Phase 3: Handcrafted image features
  Phase 4: Multimodal fusion with SelectKBest (supervised feature selection)
  Phase 5: Consensus Tri-Training Pseudo-Labeling
  Phase 6: Final OOF evaluation with Linear SVM

Target: 70%+ OOF Accuracy on 129 labeled prescription images
"""

import os, glob, warnings, random, json, datetime, sys, time
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageEnhance
from pathlib import Path
from collections import Counter
from copy import deepcopy

from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

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
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('=' * 60)
print('  BURNOUT AI — V5 FINE-TUNED MULTIMODAL PIPELINE')
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
NUM_CLASSES = len(le.classes_)
print(f'  Class mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}')


# ══════════════════════════════════════════════════════════════
# PHASE 1 — END-TO-END VISION FINE-TUNING (PyTorch)
# ══════════════════════════════════════════════════════════════
print('\n' + '─' * 60)
print('  PHASE 1 — END-TO-END VISION FINE-TUNING')
print('─' * 60)

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print('  ⚠ timm not installed, using torchvision ResNet50')

IMGSIZE = 224

# ── Transforms ────────────────────────────────────────────
train_transform = T.Compose([
    T.Resize((IMGSIZE + 32, IMGSIZE + 32)),
    T.RandomCrop(IMGSIZE),
    T.RandomHorizontalFlip(p=0.1),
    T.RandomRotation(degrees=8),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
    T.RandomAffine(degrees=5, translate=(0.08, 0.08), scale=(0.9, 1.1)),
    T.RandomGrayscale(p=0.1),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = T.Compose([
    T.Resize((IMGSIZE, IMGSIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class LabeledDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
        except Exception:
            img = Image.new('RGB', (IMGSIZE, IMGSIZE), 128)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


class UnlabeledDataset(Dataset):
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


# ── MixUp ─────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.4):
    """MixUp augmentation: blends images and labels."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Build Fine-Tunable Model ─────────────────────────────
class BurnoutClassifier(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super().__init__()
        if HAS_TIMM:
            self.backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
            self.feat_dim = self.backbone.num_features
        else:
            import torchvision.models as models
            _resnet = models.resnet50(weights='IMAGENET1K_V2')
            self.backbone = nn.Sequential(*list(_resnet.children())[:-1], nn.Flatten())
            self.feat_dim = 2048

        # Freeze everything first
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Unfreeze last 2 blocks (for EfficientNet-B3, blocks 5-6)
        if HAS_TIMM:
            unfrozen = 0
            for name, param in self.backbone.named_parameters():
                # Unfreeze blocks.5, blocks.6, conv_head, bn2
                if any(k in name for k in ['blocks.5', 'blocks.6', 'conv_head', 'bn2']):
                    param.requires_grad = True
                    unfrozen += 1
            print(f'  ✓ Unfrozen {unfrozen} parameters in last 2 blocks')
        else:
            # Unfreeze layer3 and layer4 for ResNet
            for name, param in self.backbone.named_parameters():
                if 'layer3' in name or 'layer4' in name:
                    param.requires_grad = True

        # Classification head with strong regularization
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


# ── Fine-Tuning Loop ─────────────────────────────────────
EPOCHS = 35
LR = 3e-4
WEIGHT_DECAY = 1e-3

model = BurnoutClassifier(NUM_CLASSES, 0).to(DEVICE)
feat_dim = model.feat_dim
print(f'  ✓ Backbone   : {"EfficientNet-B3" if HAS_TIMM else "ResNet-50"}')
print(f'  ✓ Feature dim: {feat_dim}')
print(f'  ✓ MixUp alpha: 0.4')
print(f'  ✓ Epochs     : {EPOCHS}')
print(f'  ✓ LR         : {LR}')

# Use class weights to handle imbalance
class_counts = df['label_enc'].value_counts().sort_index().values
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
print(f'  ✓ Class weights: {class_weights.round(3)}')

all_paths = df['full_path'].tolist()
all_labels = df['label_enc'].values

# Train on ALL labeled data (we'll evaluate via OOF later)
train_ds = LabeledDataset(all_paths, all_labels, transform=train_transform)
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

print(f'\n  Training on {len(train_ds)} images...')
t0 = time.time()
best_loss = float('inf')
patience_counter = 0

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    n_batches = 0
    for imgs, labels in train_dl:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        # Apply MixUp 50% of the time
        use_mixup = random.random() < 0.5
        if use_mixup:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.4)
            outputs = model(imgs)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    scheduler.step()
    avg_loss = epoch_loss / max(n_batches, 1)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f'    Epoch {epoch+1:2d}/{EPOCHS} — Loss: {avg_loss:.4f}  LR: {scheduler.get_last_lr()[0]:.6f}')

    # Early stopping check
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        best_state = deepcopy(model.state_dict())
    else:
        patience_counter += 1
        if patience_counter >= 8:
            print(f'    ⚠ Early stopping at epoch {epoch+1}')
            break

# Load best weights
model.load_state_dict(best_state)
model.eval()
print(f'  ⏱ Fine-tuning took {time.time()-t0:.1f}s')
print(f'  ✓ Best training loss: {best_loss:.4f}')


# ── Extract Fine-Tuned Vision Features ───────────────────
def extract_finetuned_features(paths, desc='Extracting'):
    ds = UnlabeledDataset(paths, transform=val_transform)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, pin_memory=True)
    feats = []
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(DEVICE)
            out = model(batch, return_features=True)
            feats.append(out.cpu().numpy())
    result = np.vstack(feats)
    print(f'  {desc}: {len(paths)} images → {result.shape}')
    return result


print('\n  Extracting fine-tuned vision features...')
X_vision_labeled = extract_finetuned_features(all_paths, desc='Labeled')


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
        import re
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

    print('  Extracting OCR text from labeled images...')
    nlp_features_list = []
    ocr_texts = []
    for i, path in enumerate(all_paths):
        if (i+1) % 20 == 0 or i == 0:
            print(f'    OCR: {i+1}/{len(all_paths)}', end='\r')
        text, conf, n_det = extract_text_from_image(path)
        ocr_texts.append(text)
        nlp_features_list.append(extract_nlp_features(text, conf, n_det))
    print(f'    OCR: {len(all_paths)}/{len(all_paths)} ✓         ')
    X_nlp_labeled = np.vstack(nlp_features_list)
    print(f'  ✓ NLP features shape: {X_nlp_labeled.shape}')
else:
    X_nlp_labeled = np.zeros((len(df), 1))
    print('  ⚠ Using dummy NLP features (no OCR)')


# ══════════════════════════════════════════════════════════════
# PHASE 3 — Handcrafted Image Features
# ══════════════════════════════════════════════════════════════
print('\n' + '─' * 60)
print('  PHASE 3 — HANDCRAFTED IMAGE FEATURES')
print('─' * 60)

def extract_handcrafted_features(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
    except Exception:
        return np.zeros(40)

    img = img.resize((224, 224))
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
    orig = Image.open(img_path)
    ow, oh = orig.size
    features.append(ow / (oh + 1e-8))
    features.append(float(ow * oh))
    features.append(float(ow))
    features.append(float(oh))

    # Color channel stats (4)
    for ch in range(3):
        features.append(arr[:,:,ch].mean() / 255.0)
    features.append(arr.std() / 255.0)

    return np.array(features, dtype=np.float32)


print('  Extracting handcrafted features...')
hc_features = []
for i, path in enumerate(all_paths):
    if (i+1) % 50 == 0 or i == 0:
        print(f'    Handcrafted: {i+1}/{len(all_paths)}', end='\r')
    hc_features.append(extract_handcrafted_features(path))
print(f'    Handcrafted: {len(all_paths)}/{len(all_paths)} ✓         ')
X_hc_labeled = np.vstack(hc_features)
print(f'  ✓ Handcrafted features shape: {X_hc_labeled.shape}')


# ══════════════════════════════════════════════════════════════
# PHASE 4 — MULTIMODAL FUSION + SELECTKBEST
# ══════════════════════════════════════════════════════════════
print('\n' + '─' * 60)
print('  PHASE 4 — MULTIMODAL FUSION + FEATURE SELECTION')
print('─' * 60)

y_labeled = df['label_enc'].values

# Concatenate all feature modalities
X_raw = np.hstack([X_vision_labeled, X_nlp_labeled, X_hc_labeled])
print(f'  ✓ Vision features  : {X_vision_labeled.shape[1]}')
print(f'  ✓ NLP features     : {X_nlp_labeled.shape[1]}')
print(f'  ✓ Handcrafted      : {X_hc_labeled.shape[1]}')
print(f'  ✓ TOTAL fused (raw): {X_raw.shape[1]} dimensions')

# Replace NaN/Inf with 0
X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

# Supervised feature selection — SelectKBest with ANOVA F-value
K_FEATURES = min(300, X_raw.shape[1], X_raw.shape[0] - 1)
selector = SelectKBest(f_classif, k=K_FEATURES)
X_labeled = selector.fit_transform(X_raw, y_labeled)
print(f'  ✓ SelectKBest      : {X_raw.shape[1]} → {X_labeled.shape[1]} features')
print(f'  ✓ Labeled samples  : {X_labeled.shape[0]}')

# Show which feature blocks survived
selected_mask = selector.get_support()
vision_kept = selected_mask[:X_vision_labeled.shape[1]].sum()
nlp_kept = selected_mask[X_vision_labeled.shape[1]:X_vision_labeled.shape[1]+X_nlp_labeled.shape[1]].sum()
hc_kept = selected_mask[X_vision_labeled.shape[1]+X_nlp_labeled.shape[1]:].sum()
print(f'  ✓ Vision kept      : {vision_kept}/{X_vision_labeled.shape[1]}')
print(f'  ✓ NLP kept         : {nlp_kept}/{X_nlp_labeled.shape[1]}')
print(f'  ✓ Handcrafted kept : {hc_kept}/{X_hc_labeled.shape[1]}')


# ══════════════════════════════════════════════════════════════
# PHASE 4.5 — BASELINE EVALUATION (before pseudo-labeling)
# ══════════════════════════════════════════════════════════════
print('\n' + '─' * 60)
print('  PHASE 4.5 — BASELINE CLASSIFIERS (pre-pseudo-labeling)')
print('─' * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Linear SVM
print('\n  Testing Linear SVM...')
lsvm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', LinearSVC(C=1.0, class_weight='balanced', max_iter=5000, random_state=SEED))
])
lsvm_acc = cross_val_score(lsvm_pipe, X_labeled, y_labeled, cv=skf, scoring='accuracy', n_jobs=1)
lsvm_f1  = cross_val_score(lsvm_pipe, X_labeled, y_labeled, cv=skf, scoring='f1_macro', n_jobs=1)
print(f'  📊 LSVM — Acc: {lsvm_acc.mean()*100:.1f}% ± {lsvm_acc.std()*100:.1f}%  |  F1: {lsvm_f1.mean():.4f}')

# RBF SVM
print('  Testing RBF SVM...')
rbf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale',
                class_weight='balanced', probability=True, random_state=SEED))
])
rbf_acc = cross_val_score(rbf_pipe, X_labeled, y_labeled, cv=skf, scoring='accuracy', n_jobs=1)
rbf_f1  = cross_val_score(rbf_pipe, X_labeled, y_labeled, cv=skf, scoring='f1_macro', n_jobs=1)
print(f'  📊 RBF  — Acc: {rbf_acc.mean()*100:.1f}% ± {rbf_acc.std()*100:.1f}%  |  F1: {rbf_f1.mean():.4f}')

# Random Forest
print('  Testing Random Forest...')
rf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=3,
        class_weight='balanced', random_state=SEED, n_jobs=1
    ))
])
rf_acc = cross_val_score(rf_pipe, X_labeled, y_labeled, cv=skf, scoring='accuracy', n_jobs=1)
rf_f1  = cross_val_score(rf_pipe, X_labeled, y_labeled, cv=skf, scoring='f1_macro', n_jobs=1)
print(f'  📊 RF   — Acc: {rf_acc.mean()*100:.1f}% ± {rf_acc.std()*100:.1f}%  |  F1: {rf_f1.mean():.4f}')

# XGBoost
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
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=2.0,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=SEED, n_jobs=1
        ))
    ])
    xgb_acc = cross_val_score(xgb_pipe, X_labeled, y_labeled, cv=skf, scoring='accuracy', n_jobs=1)
    xgb_f1  = cross_val_score(xgb_pipe, X_labeled, y_labeled, cv=skf, scoring='f1_macro', n_jobs=1)
    print(f'  📊 XGB  — Acc: {xgb_acc.mean()*100:.1f}% ± {xgb_acc.std()*100:.1f}%  |  F1: {xgb_f1.mean():.4f}')


# ══════════════════════════════════════════════════════════════
# PHASE 5 — CONSENSUS TRI-TRAINING PSEUDO-LABELING
# ══════════════════════════════════════════════════════════════
print('\n' + '─' * 60)
print('  PHASE 5 — CONSENSUS TRI-TRAINING')
print('─' * 60)

# Extract features for unlabeled images
print('  Extracting features from unlabeled images...')
t0 = time.time()
X_vision_unlabeled = extract_finetuned_features(unlabeled_imgs, desc='Unlabeled vision')

print('  Extracting handcrafted features from unlabeled images...')
hc_unlabeled = []
for i, path in enumerate(unlabeled_imgs):
    if (i+1) % 100 == 0 or i == 0:
        print(f'    Handcrafted: {i+1}/{len(unlabeled_imgs)}', end='\r')
    hc_unlabeled.append(extract_handcrafted_features(path))
print(f'    Handcrafted: {len(unlabeled_imgs)}/{len(unlabeled_imgs)} ✓         ')
X_hc_unlabeled = np.vstack(hc_unlabeled)

if HAS_OCR:
    print('  Extracting OCR from unlabeled images...')
    nlp_unlabeled = []
    for i, path in enumerate(unlabeled_imgs):
        if (i+1) % 50 == 0 or i == 0:
            print(f'    OCR: {i+1}/{len(unlabeled_imgs)}', end='\r')
        text, conf, n_det = extract_text_from_image(path)
        nlp_unlabeled.append(extract_nlp_features(text, conf, n_det))
    print(f'    OCR: {len(unlabeled_imgs)}/{len(unlabeled_imgs)} ✓         ')
    X_nlp_unlabeled = np.vstack(nlp_unlabeled)
else:
    X_nlp_unlabeled = np.zeros((len(unlabeled_imgs), X_nlp_labeled.shape[1]))

print(f'  ⏱ Unlabeled feature extraction took {time.time()-t0:.1f}s')

# ── Tri-Training: 3 independent models on different modalities ──
print('\n  Training 3 modal-specific classifiers...')

# Scale each modality to avoid dominant features
scaler_v = StandardScaler().fit(X_vision_labeled)
scaler_n = StandardScaler().fit(X_nlp_labeled)
scaler_h = StandardScaler().fit(X_hc_labeled)

Xv_tr = scaler_v.transform(X_vision_labeled)
Xn_tr = scaler_n.transform(X_nlp_labeled)
Xh_tr = scaler_h.transform(X_hc_labeled)

Xv_un = scaler_v.transform(X_vision_unlabeled)
Xn_un = scaler_n.transform(X_nlp_unlabeled)
Xh_un = scaler_h.transform(X_hc_unlabeled)

# Model A: Vision-only SVM
model_a = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced',
              probability=True, random_state=SEED)
model_a.fit(Xv_tr, y_labeled)
preds_a = model_a.predict(Xv_un)
probs_a = model_a.predict_proba(Xv_un)
print(f'  ✓ Model A (Vision SVM) trained')

# Model B: NLP-only Random Forest
model_b = RandomForestClassifier(
    n_estimators=300, max_depth=6, class_weight='balanced',
    random_state=SEED, n_jobs=1
)
model_b.fit(Xn_tr, y_labeled)
preds_b = model_b.predict(Xn_un)
probs_b = model_b.predict_proba(Xn_un)
print(f'  ✓ Model B (NLP RF) trained')

# Model C: Handcrafted-only SVM
model_c = SVC(kernel='rbf', C=10, gamma='scale', class_weight='balanced',
              probability=True, random_state=SEED)
model_c.fit(Xh_tr, y_labeled)
preds_c = model_c.predict(Xh_un)
probs_c = model_c.predict_proba(Xh_un)
print(f'  ✓ Model C (Handcrafted SVM) trained')

# ── Consensus Voting ─────────────────────────────────────
# Accept pseudo-label ONLY if all 3 models agree AND avg confidence > threshold
CONF_THRESHOLD = 0.55  # lower threshold since we require 3-way agreement
consensus_mask = (preds_a == preds_b) & (preds_b == preds_c)
avg_conf = (probs_a.max(axis=1) + probs_b.max(axis=1) + probs_c.max(axis=1)) / 3
combined_mask = consensus_mask & (avg_conf >= CONF_THRESHOLD)

X_unlabeled_raw = np.hstack([X_vision_unlabeled, X_nlp_unlabeled, X_hc_unlabeled])
X_unlabeled_raw = np.nan_to_num(X_unlabeled_raw, nan=0.0, posinf=0.0, neginf=0.0)
X_unlabeled_selected = selector.transform(X_unlabeled_raw)

X_pseudo = X_unlabeled_selected[combined_mask]
y_pseudo = preds_a[combined_mask]  # all 3 agree, so any one works

print(f'\n  📊 Consensus Tri-Training Results:')
print(f'    Total unlabeled              : {len(unlabeled_imgs)}')
print(f'    3-way agreement              : {consensus_mask.sum()} ({consensus_mask.mean()*100:.1f}%)')
print(f'    + confidence >= {CONF_THRESHOLD}        : {combined_mask.sum()} ({combined_mask.mean()*100:.1f}%)')
for i, cls in enumerate(le.classes_):
    n = (y_pseudo == i).sum()
    print(f'      {cls:8s} : {n}')

# ── Combine ──────────────────────────────────────────────
X_combined = np.vstack([X_labeled, X_pseudo])
y_combined = np.concatenate([y_labeled, y_pseudo])
print(f'\n  ✓ Combined dataset : {len(X_combined)} images ({len(X_labeled)} real + {len(X_pseudo)} pseudo)')


# ══════════════════════════════════════════════════════════════
# PHASE 6 — FINAL OOF EVALUATION
# ══════════════════════════════════════════════════════════════
print('\n' + '─' * 60)
print('  PHASE 6 — FINAL EVALUATION (OOF)')
print('─' * 60)

classifiers = {}

classifiers['LinearSVM'] = lambda: Pipeline([
    ('scaler', StandardScaler()),
    ('svm', LinearSVC(C=1.0, class_weight='balanced', max_iter=5000, random_state=SEED))
])

classifiers['RBF_SVM'] = lambda: Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale',
                class_weight='balanced', probability=True, random_state=SEED))
])

classifiers['RBF_SVM_C50'] = lambda: Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=50, gamma='auto',
                class_weight='balanced', probability=True, random_state=SEED))
])

classifiers['RF'] = lambda: Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(
        n_estimators=500, max_depth=8, min_samples_leaf=3,
        class_weight='balanced', random_state=SEED, n_jobs=1
    ))
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

best_oof_acc = 0
best_oof_f1 = 0
best_name = ''
best_oof_preds = None

for clf_name, clf_factory in classifiers.items():
    print(f'\n  Evaluating {clf_name} (OOF on labeled, trained on combined)...')
    oof_preds = np.zeros(len(X_labeled), dtype=int)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_labeled, y_labeled)):
        clf = clf_factory()
        X_tr = np.vstack([X_labeled[tr_idx], X_pseudo])
        y_tr = np.concatenate([y_labeled[tr_idx], y_pseudo])
        clf.fit(X_tr, y_tr)

        if hasattr(clf, 'predict'):
            oof_preds[val_idx] = clf.predict(X_labeled[val_idx])

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

final_model = classifiers[best_name]()
final_model.fit(X_combined, y_combined)

ts = datetime.datetime.now().strftime('%Y%m%d_%H%M')
model_path   = SAVE_DIR / f'burnout_v5_{best_name}_{ts}.pkl'
encoder_path = SAVE_DIR / 'label_encoder_v5.pkl'

joblib.dump(final_model, model_path)
joblib.dump(le, encoder_path)
joblib.dump(final_model, SAVE_DIR / 'burnout_v5_latest.pkl')
joblib.dump(selector, SAVE_DIR / 'feature_selector_v5.pkl')

# Save the fine-tuned PyTorch backbone
torch.save(model.state_dict(), str(SAVE_DIR / 'backbone_v5.pth'))

print(f'  ✓ Model saved   : {model_path.name}')
print(f'  ✓ Latest        : burnout_v5_latest.pkl')
print(f'  ✓ Encoder       : label_encoder_v5.pkl')
print(f'  ✓ Selector      : feature_selector_v5.pkl')
print(f'  ✓ Backbone      : backbone_v5.pth')

# Save results JSON
report = classification_report(y_labeled, best_oof_preds, target_names=le.classes_, output_dict=True)

results = {
    "version": "5.0",
    "strategy": "Fine-Tuned Vision + NLP + Handcrafted + Tri-Training Pseudo-Labels",
    "backbone": "EfficientNet-B3 (fine-tuned, last 2 blocks)" if HAS_TIMM else "ResNet-50 (fine-tuned)",
    "ocr_engine": "EasyOCR" if HAS_OCR else "None",
    "classifier": best_name,
    "feature_dim_raw": int(X_raw.shape[1]),
    "feature_dim_selected": int(X_labeled.shape[1]),
    "feature_breakdown": {
        "vision_raw": int(X_vision_labeled.shape[1]),
        "vision_kept": int(vision_kept),
        "nlp_raw": int(X_nlp_labeled.shape[1]),
        "nlp_kept": int(nlp_kept),
        "handcrafted_raw": int(X_hc_labeled.shape[1]),
        "handcrafted_kept": int(hc_kept)
    },
    "labeled_images": int(len(X_labeled)),
    "pseudo_labeled": int(len(X_pseudo)),
    "total_training": int(len(X_combined)),
    "tri_training": {
        "total_unlabeled": len(unlabeled_imgs),
        "3way_agreement": int(consensus_mask.sum()),
        "confident_kept": int(combined_mask.sum()),
        "confidence_threshold": CONF_THRESHOLD
    },
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
        "V4_Multimodal": {"accuracy": 39.53, "note": "Frozen EfficientNet + PCA + XGBoost"},
        "V5_FineTuned": {"accuracy": round(best_oof_acc * 100, 2), "note": f"{best_name} + Fine-Tuned Vision + Tri-Training"}
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
print('  DOCTOR BURNOUT DETECTION — V5 RESULTS')
print('=' * 60)
print(f'  Strategy       : Fine-Tuned Multimodal + Tri-Training')
print(f'  Vision Backbone: {"EfficientNet-B3 (fine-tuned)" if HAS_TIMM else "ResNet-50 (fine-tuned)"}')
print(f'  OCR Engine     : {"EasyOCR" if HAS_OCR else "None"}')
print(f'  Classifier     : {best_name}')
print(f'  Feature Dim    : {X_raw.shape[1]} → {X_labeled.shape[1]} (SelectKBest)')
print(f'  Labeled        : {len(X_labeled)}')
print(f'  Pseudo-labeled : {len(X_pseudo)} (from {len(unlabeled_imgs)} unlabeled)')
print(f'  Total training : {len(X_combined)}')
print(f'  GPU            : {torch.cuda.get_device_name(0) if DEVICE == "cuda" else "CPU"}')
print(f'')
print(f'  OOF Accuracy   : {best_oof_acc*100:.2f}%')
print(f'  OOF Macro F1   : {best_oof_f1:.4f}')
print(f'')
print(f'  vs V1 (CNN)    : {44.2}%')
print(f'  vs V3 (HC+XGB) : {36.43}%')
print(f'  vs V4 (Frozen) : {39.53}%')
delta = (best_oof_acc - 0.3953) * 100
print(f'  IMPROVEMENT    : {"+" if delta >= 0 else ""}{delta:.1f}% over V4')
print('=' * 60)
print('\n✅ Done!')
