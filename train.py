"""
CSE 144 Final Project – Transfer Learning Image Classifier
100-class classification with ~10 training images per class.

Usage:
    python3 train.py                 # train + generate submission.csv
    python3 train.py --predict-only  # load best checkpoint, skip training
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
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models
from torchvision.models import EfficientNet_B2_Weights
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
IMG_SIZE     = 224          # EfficientNet-B2 native resolution
BATCH_SIZE   = 32
NUM_CLASSES  = 100
NUM_WORKERS  = 0            # 0 = fully reproducible on all platforms

# Two-phase training (fast enough on MPS/CPU, ~15–25 min total)
PHASE1_EPOCHS = 8           # head warm-up  (backbone frozen)
PHASE2_EPOCHS = 45          # full fine-tune (progressive LR)

PHASE1_LR    = 3e-3
PHASE2_LR    = 1e-4         # base for backbone; head gets 5× more

WEIGHT_DECAY = 1e-4
LABEL_SMOOTH = 0.1
MIXUP_ALPHA  = 0.3

# ──────────────────────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE + 24, IMG_SIZE + 24)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),   # automated augmentation policy
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
# Model – EfficientNet-B2  (7.8M params, fast on MPS)
# ──────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    # Slightly wider head with two FC layers for more capacity
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.35, inplace=True),
        nn.Linear(in_features, 512),
        nn.SiLU(inplace=True),
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


# ──────────────────────────────────────────────────────────────────────────────
# MixUp
# ──────────────────────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha: float = MIXUP_ALPHA):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1.0 - lam) * x[idx], y, y[idx], lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)


# ──────────────────────────────────────────────────────────────────────────────
# Training / evaluation
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scheduler=None, use_mixup=True):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in tqdm(loader, leave=False, desc="  train"):
        imgs, labels = imgs.to(device), labels.to(device)

        if use_mixup:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels)
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
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total


# ──────────────────────────────────────────────────────────────────────────────
# Test-time augmentation (5 views → average softmax)
# ──────────────────────────────────────────────────────────────────────────────

TTA_TF_LIST = [
    EVAL_TF,
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 24, IMG_SIZE + 24)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE + 24, IMG_SIZE + 24)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
    transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]),
]


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
    print(f"Model: EfficientNet-B2  |  {n_all:,} total params")

    if predict_only:
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded checkpoint: epoch {ckpt['epoch']}  val_acc {ckpt.get('val_acc', '?'):.4f}")
    else:
        # 5 % validation split (keeps ~54 images for checkpoint selection)
        ds_aug = TrainDataset(TRAIN_DIR, transform=TRAIN_TF)
        ds_cln = TrainDataset(TRAIN_DIR, transform=EVAL_TF)
        n      = len(ds_aug)
        n_val  = max(1, int(n * 0.05))
        gen    = torch.Generator().manual_seed(SEED)
        idx    = torch.randperm(n, generator=gen).tolist()
        train_ds = Subset(ds_aug, idx[n_val:])
        val_ds   = Subset(ds_cln, idx[:n_val])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True,  num_workers=NUM_WORKERS)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=NUM_WORKERS)
        print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

        criterion    = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
        best_val_acc = 0.0

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
            v_loss,  v_acc  = evaluate(model, val_loader, criterion)
            print(f"  Ep {ep:2d}/{PHASE1_EPOCHS} | "
                  f"tr {tr_loss:.3f}/{tr_acc:.3f} | val {v_loss:.3f}/{v_acc:.3f}")
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                torch.save({"model_state_dict": model.state_dict(),
                            "epoch": ep, "val_acc": v_acc}, CKPT_PATH)

        # ── Phase 2: full fine-tune with layer-wise LR ────────────────────
        unfreeze_all(model)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n── Phase 2 ({PHASE2_EPOCHS} ep, full fine-tune, {n_trainable:,} trainable) ──")

        # Give the head a higher LR than the backbone (3× ratio)
        classifier_ids = {id(p) for p in model.classifier.parameters()}
        backbone_params   = [p for p in model.parameters() if id(p) not in classifier_ids]
        classifier_params = [p for p in model.classifier.parameters()]

        opt2 = torch.optim.AdamW([
            {"params": backbone_params,   "lr": PHASE2_LR},
            {"params": classifier_params, "lr": PHASE2_LR * 3},
        ], weight_decay=WEIGHT_DECAY)

        # Cosine decay with short linear warm-up
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
                                              use_mixup=True)
            v_loss,  v_acc  = evaluate(model, val_loader, criterion)
            print(f"  Ep {ep:2d}/{PHASE2_EPOCHS} | "
                  f"tr {tr_loss:.3f}/{tr_acc:.3f} | val {v_loss:.3f}/{v_acc:.3f}")
            if v_acc > best_val_acc:
                best_val_acc = v_acc
                torch.save({"model_state_dict": model.state_dict(),
                            "epoch": PHASE1_EPOCHS + ep, "val_acc": v_acc}, CKPT_PATH)
                print(f"  ✓ New best: {best_val_acc:.4f}")

        print(f"\nDone. Best val acc: {best_val_acc:.4f}")
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best checkpoint from epoch {ckpt['epoch']}")

    # ── 5-view TTA inference ───────────────────────────────────────────────
    print("\nRunning 5-view TTA inference ...")
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
