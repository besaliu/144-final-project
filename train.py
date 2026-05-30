"""
CSE 144 Final Project – Transfer Learning Image Classifier
100-class classification with ~10 training images per class.

Usage:
    python3 train.py                 # train + generate submission.csv
    python3 train.py --predict-only  # load best checkpoint, skip training

Requires: pip install timm
"""

import os
import ssl
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# macOS Python.org installs lack SSL certs – use certifi bundle for downloads
try:
    import certifi
    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where()
    )
except ImportError:
    pass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
from tqdm.auto import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent / "ucsc-cse-144-spring-2026-final-project"
TRAIN_DIR  = BASE_DIR / "train"
TEST_DIR   = BASE_DIR / "test"
SAMPLE_CSV = BASE_DIR / "sample_submission.csv"
CKPT_DIR   = Path(__file__).parent / "checkpoints"
CKPT_PATH  = CKPT_DIR / "best_model.pt"
OUTPUT_CSV = Path(__file__).parent / "submission.csv"

CKPT_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42

def set_seed(seed: int = SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed()

device = (
    "cuda" if torch.cuda.is_available()
    else "mps"  if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}  |  torch {torch.__version__}")

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
IMG_SIZE     = 224
BATCH_SIZE   = 32
NUM_CLASSES  = 100
NUM_WORKERS  = 0            # 0 = fully reproducible on all platforms

PHASE1_EPOCHS = 8           # head warm-up  (backbone frozen)
PHASE2_EPOCHS = 55          # full fine-tune (all layers, cosine decay)

PHASE1_LR    = 3e-3
PHASE2_LR    = 1e-4         # backbone base LR; head gets 8× more

WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1
MIXUP_ALPHA  = 0.3
CUTMIX_ALPHA = 1.0

# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE + 24, IMG_SIZE + 24)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=13),  # was 9
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

EVAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ──────────────────────────────────────────────────────────────────────────────
# Datasets
# ──────────────────────────────────────────────────────────────────────────────

class TrainDataset(Dataset):
    """
    Loads images from train/<class_folder>/*.jpg.
    Folder name is the integer label (folder "k" → label k).
    """
    def __init__(self, root: Path, transform=None):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        class_dirs = sorted(
            (p for p in root.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name),
        )
        for cls_dir in class_dirs:
            label = int(cls_dir.name)
            for img_path in sorted(cls_dir.glob("*.jpg")):
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class TestDataset(Dataset):
    """Loads unlabeled test images; returns (image_tensor, filename)."""
    def __init__(self, test_dir: Path, file_ids: list[str], transform=None):
        self.test_dir  = test_dir
        self.file_ids  = file_ids
        self.transform = transform

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        fname = self.file_ids[idx]
        img   = Image.open(self.test_dir / fname).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, fname


# ──────────────────────────────────────────────────────────────────────────────
# EMA
# ──────────────────────────────────────────────────────────────────────────────

class EMA:
    """Maintains a shadow copy of model weights as an exponential moving average.
    With ~1870 total Phase-2 steps, decay=0.999 gives an effective window of
    ~1000 steps (≈30 epochs), smoothing late-training noise without over-lagging.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.shadow = {k: v.clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1.0 - self.decay)

    def apply(self, model: nn.Module):
        """Swap model weights to EMA weights; stash originals for restore()."""
        self._stash = {k: v.clone() for k, v in model.state_dict().items()}
        ema_state   = {k: v.to(v.dtype).to(next(model.parameters()).device)
                       for k, v in self.shadow.items()}
        model.load_state_dict(ema_state)

    def restore(self, model: nn.Module):
        model.load_state_dict(self._stash)


# ──────────────────────────────────────────────────────────────────────────────
# Model – EfficientNet-B2 with Noisy Student weights (timm)
# Same 7.8M params as torchvision B2 but pretrained on JFT-300M via self-training,
# giving substantially richer features for diverse 100-class transfer.
# ──────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = timm.create_model(
        "tf_efficientnet_b2.ns_jft_in1k",
        pretrained=True,
        num_classes=num_classes,
        drop_rate=0.0,  # we manage dropout inside our custom head
    )
    in_features = model.num_features  # 1408
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35),
        nn.Linear(in_features, 512),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module):
    for name, p in model.named_parameters():
        p.requires_grad = "classifier" in name


def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True


def make_param_groups(model: nn.Module, base_lr: float, head_lr: float) -> list[dict]:
    classifier_ids = {id(p) for p in model.classifier.parameters()}
    backbone_params   = [p for p in model.parameters()
                         if p.requires_grad and id(p) not in classifier_ids]
    classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]
    groups = []
    if backbone_params:
        groups.append({"params": backbone_params,   "lr": base_lr})
    if classifier_params:
        groups.append({"params": classifier_params, "lr": head_lr})
    return groups


# ──────────────────────────────────────────────────────────────────────────────
# MixUp + CutMix
# ──────────────────────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha: float = MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[idx], y, y[idx], lam


def cutmix_data(x, y, alpha: float = CUTMIX_ALPHA):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    H, W = x.size(2), x.size(3)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)
    cy = np.random.randint(H)
    cx = np.random.randint(W)
    y1 = max(0, cy - cut_h // 2)
    y2 = min(H, cy + cut_h // 2)
    x1 = max(0, cx - cut_w // 2)
    x2 = min(W, cx + cut_w // 2)
    x_mixed = x.clone()
    x_mixed[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    return x_mixed, y, y[idx], lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scheduler=None, use_mixup=True, ema=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, leave=False, desc="  train"):
        imgs, labels = imgs.to(device), labels.to(device)

        if use_mixup:
            # Alternate randomly between MixUp and CutMix each batch
            if random.random() < 0.5:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels)
            else:
                imgs, y_a, y_b, lam = cutmix_data(imgs, labels)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


# ──────────────────────────────────────────────────────────────────────────────
# Test-time augmentation – 10 diverse views → averaged softmax
# ──────────────────────────────────────────────────────────────────────────────

def _build_tta_transforms() -> list:
    S  = IMG_SIZE
    SL = IMG_SIZE + 32  # 256 for 224
    SS = int(IMG_SIZE * 0.875)  # 196 for 224
    pad = (S - SS) // 2         # 14 → 196 + 28 = 224
    return [
        # 1. Standard center
        transforms.Compose([
            transforms.Resize((S, S)),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        # 2. HFlip
        transforms.Compose([
            transforms.Resize((S, S)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        # 3. Larger resize + center crop (zoom-out effect)
        transforms.Compose([
            transforms.Resize((SL, SL)),
            transforms.CenterCrop(S),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        # 4. Larger resize + center crop + HFlip
        transforms.Compose([
            transforms.Resize((SL, SL)),
            transforms.CenterCrop(S),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        # 5. Slight color jitter
        transforms.Compose([
            transforms.Resize((S, S)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        # 6. Rotation +10°
        transforms.Compose([
            transforms.Resize((S, S)),
            transforms.RandomRotation(degrees=(10, 10)),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        # 7. Rotation −10°
        transforms.Compose([
            transforms.Resize((S, S)),
            transforms.RandomRotation(degrees=(-10, -10)),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        # 8. Smaller object (resize small + pad to S, zoom-in effect)
        transforms.Compose([
            transforms.Resize((SS, SS)),
            transforms.Pad(pad),
            transforms.CenterCrop(S),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        # 9. HFlip + color jitter
        transforms.Compose([
            transforms.Resize((S, S)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        # 10. Larger resize + rotation + center crop
        transforms.Compose([
            transforms.Resize((SL, SL)),
            transforms.RandomRotation(degrees=(-8, 8)),
            transforms.CenterCrop(S),
            transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
    ]

TTA_TF_LIST = _build_tta_transforms()


@torch.no_grad()
def predict(model, file_ids: list[str], tta: bool = True) -> dict[str, int]:
    model.eval()
    tf_list   = TTA_TF_LIST if tta else [EVAL_TF]
    all_probs: dict[str, np.ndarray] = {}

    for tf in tf_list:
        ds     = TestDataset(TEST_DIR, file_ids, transform=tf)
        loader = DataLoader(ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)
        for imgs, fnames in tqdm(loader, leave=False, desc="  predict"):
            probs = torch.softmax(model(imgs.to(device)), dim=1).cpu().numpy()
            for fname, p in zip(fnames, probs):
                if fname not in all_probs:
                    all_probs[fname] = np.zeros(NUM_CLASSES, dtype=np.float64)
                all_probs[fname] += p

    return {fname: int(np.argmax(p)) for fname, p in all_probs.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(predict_only: bool = False):
    set_seed()

    sample_df = pd.read_csv(SAMPLE_CSV)
    extra_ids = sorted(
        [f.name for f in TEST_DIR.glob("*.jpg")
         if f.name not in set(sample_df["ID"].tolist())],
        key=lambda x: int(x.split(".")[0]),
    )
    all_ids = sample_df["ID"].tolist() + extra_ids
    print(f"Test images: {len(all_ids)} "
          f"({len(sample_df)} from sample + {len(extra_ids)} extra)")

    model = build_model().to(device)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"Model: EfficientNet-B2 (ns_jft_in1k)  |  {n_all:,} total params")

    if predict_only:
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: epoch {ckpt['epoch']}  "
              f"train_acc {ckpt.get('train_acc', 0.0):.4f}")
    else:
        # Train on ALL samples – 5% holdout has <1 image/class, too noisy to trust
        train_ds     = TrainDataset(TRAIN_DIR, transform=TRAIN_TF)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=NUM_WORKERS)
        print(f"Train: {len(train_ds)} samples (full dataset, no holdout)")

        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

        # ── Phase 1: head warm-up (backbone frozen) ────────────────────────
        freeze_backbone(model)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n── Phase 1 ({PHASE1_EPOCHS} ep, head only, {n_trainable:,} trainable) ──")

        opt1 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=PHASE1_LR, weight_decay=WEIGHT_DECAY,
        )
        sch1 = torch.optim.lr_scheduler.OneCycleLR(
            opt1, max_lr=PHASE1_LR,
            total_steps=PHASE1_EPOCHS * len(train_loader), pct_start=0.2,
        )

        for ep in range(1, PHASE1_EPOCHS + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, opt1, criterion, sch1,
                                              use_mixup=False)
            print(f"  Ep {ep:2d}/{PHASE1_EPOCHS} | tr {tr_loss:.3f}/{tr_acc:.3f}")

        # ── Phase 2: full fine-tune, single optimizer, cosine decay ───────
        unfreeze_all(model)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n── Phase 2 ({PHASE2_EPOCHS} ep, full fine-tune, {n_trainable:,} trainable) ──")

        # Head LR = 8× backbone — backbone needs gentle updates, head adapts faster
        param_groups = make_param_groups(model, PHASE2_LR, PHASE2_LR * 8)
        opt2 = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
        ema  = EMA(model, decay=0.999)

        total_steps2 = PHASE2_EPOCHS * len(train_loader)
        warmup_steps = 2 * len(train_loader)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps2 - warmup_steps)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        sch2 = torch.optim.lr_scheduler.LambdaLR(opt2, lr_lambda)

        for ep in range(1, PHASE2_EPOCHS + 1):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, opt2, criterion, sch2,
                                              use_mixup=True, ema=ema)
            print(f"  Ep {ep:2d}/{PHASE2_EPOCHS} | tr {tr_loss:.3f}/{tr_acc:.3f}")

        # Apply EMA weights before saving — smoother than raw final-epoch weights
        ema.apply(model)
        torch.save({"model_state_dict": model.state_dict(),
                    "epoch": PHASE1_EPOCHS + PHASE2_EPOCHS,
                    "train_acc": tr_acc}, CKPT_PATH)
        ema.restore(model)
        print(f"\nDone. Saved EMA checkpoint (epoch {PHASE1_EPOCHS + PHASE2_EPOCHS})")

    # ── 10-view TTA inference ──────────────────────────────────────────────
    print("\nRunning 10-view TTA inference ...")
    predictions = predict(model, all_ids, tta=True)

    rows = [{"ID": fid, "Label": predictions[fid]} for fid in all_ids]
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
    print(f"Saved → {OUTPUT_CSV}  ({len(rows)} rows)")
    print(pd.DataFrame(rows).head(10).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict-only", action="store_true",
                        help="Skip training; load best checkpoint and generate submission")
    args = parser.parse_args()
    main(predict_only=args.predict_only)
