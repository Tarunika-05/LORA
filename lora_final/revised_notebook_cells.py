# %% [markdown]
# # Parameter-Efficient LoRA Fine-Tuning for Enhanced Marine Debris Detection
# ### Revised Implementation with Full Ablation Studies
#
# **Changes from v1:**
# - Multi-seed training (3 seeds) with mean ± std reporting
# - Learning rate ablation for fair comparison
# - Decoder trainability ablation
# - Target module ablation
# - Test set evaluation on held-out 359 patches
# - Inference time and GPU memory reporting
# - Per-class IoU analysis
# - Qualitative visualizations
#
# **Conv-LoRA References:**
# - FacT: Factor-Tuning for Lightweight Adaptation (ICLR 2024)
# - Yeh et al., "Navigating Text-to-Image Customization" (Conv-LoRA, ICLR 2024)
# - Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)

# %% [markdown]
# ## Cell 1: Setup & Dependencies

# %%
print("\n" + "="*80)
print("SETUP: DEPENDENCIES & DATA")
print("="*80)

# Install without overwriting Kaggle's pre-installed PyTorch
!pip install -q transformers peft timm accelerate
!pip install -q segmentation-models-pytorch --no-deps
!pip install -q pretrainedmodels efficientnet-pytorch

# Clone repo and add to path (Kaggle doesn't support %cd)
import sys
!git clone https://github.com/harshini2806/lora_paper.git /kaggle/working/lora_paper
sys.path.insert(0, '/kaggle/working/lora_paper')

import os, time, json, random, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from typing import Dict, Any, List, Optional, Tuple
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from segmentation_models_pytorch import Unet, FPN, UnetPlusPlus

# Constants
SEEDS = [42, 123, 456]  # 3 seeds for statistical rigor
BATCH_SIZE = 4
IMG_SIZE = 224
NUM_CLASSES = 11
DEBRIS_ID = 0
EPOCHS = 70
ABLATION_EPOCHS = 50  # Faster sweeps for hyperparameter search; final evals use EPOCHS
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Kaggle: /kaggle/working/ is persisted as notebook output
# To resume across sessions, save checkpoints + results as a Kaggle Dataset
SAVE_DIR = '/kaggle/working/lora_experiments_v2'
os.makedirs(SAVE_DIR, exist_ok=True)

CLASS_NAMES = [
    'Marine Debris', 'Dense Sargassum', 'Sparse Sargassum',
    'Natural Organic Material', 'Ship', 'Clouds',
    'Marine Water', 'Sediment-Laden Water', 'Foam',
    'Turbid Water', 'Shallow Water'
]

# Data Loading
from dataloader import GenDEBRIS
# NOTE: Update DATA_PATH if you uploaded the dataset separately as a Kaggle Dataset
DATA_PATH = '/kaggle/working/lora_paper/data'
train_ds = GenDEBRIS(mode='train', path=DATA_PATH)
val_ds = GenDEBRIS(mode='val', path=DATA_PATH)
test_ds = GenDEBRIS(mode='test', path=DATA_PATH)
print(f"Dataset: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

# Class weights (inverse frequency)
label_counts = Counter()
for _, mask in train_ds:
    valid = mask[mask != -1].flatten()
    label_counts.update(valid.tolist())

total_pixels = sum(label_counts.values())
freqs = np.array([label_counts.get(i, 0) / total_pixels for i in range(NUM_CLASSES)])
class_weights = torch.tensor(1.0 / (freqs + 1e-6), dtype=torch.float32).to(device)
class_weights /= class_weights.sum()

print("SETUP COMPLETE")

# %% [markdown]
# ## Cell 2: Metrics & Inference Timing

# %%
def compute_metrics(preds, labels, num_classes=NUM_CLASSES):
    if preds.dim() == 4:
        preds = torch.argmax(preds, dim=1).cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    valid = labels != -1
    preds_v, labels_v = preds[valid], labels[valid]

    accuracy = (preds_v == labels_v).sum() / labels_v.size
    f1 = f1_score(labels_v, preds_v, average='weighted', zero_division=0)

    present_classes = np.unique(labels_v)
    ious = []
    for cls in present_classes:
        inter = ((preds_v == cls) & (labels_v == cls)).sum()
        union = ((preds_v == cls) | (labels_v == cls)).sum()
        ious.append(inter / union if union > 0 else 0.0)
    mean_iou = np.mean(ious)

    debris_inter = ((labels_v == DEBRIS_ID) & (preds_v == DEBRIS_ID)).sum()
    debris_union = ((labels_v == DEBRIS_ID) | (preds_v == DEBRIS_ID)).sum()
    debris_iou = debris_inter / debris_union if debris_union > 0 else 0.0

    return {'Accuracy': float(accuracy), 'F1': float(f1),
            'mIoU': float(mean_iou), 'Debris_IoU': float(debris_iou)}


def compute_per_class_iou(preds, labels, num_classes=NUM_CLASSES):
    """Returns IoU for each of the 11 classes."""
    if preds.dim() == 4:
        preds = torch.argmax(preds, dim=1).cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    valid = labels != -1
    preds_v, labels_v = preds[valid], labels[valid]

    ious = {}
    for cls in range(num_classes):
        inter = ((preds_v == cls) & (labels_v == cls)).sum()
        union = ((preds_v == cls) | (labels_v == cls)).sum()
        ious[CLASS_NAMES[cls]] = float(inter / union) if union > 0 else float('nan')
    return ious


def measure_inference_time(model, num_runs=50):
    """Measure inference time (ms) and GPU memory (MB)."""
    model.eval()
    dummy = torch.randn(1, 11, IMG_SIZE, IMG_SIZE).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)
    torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with autocast("cuda"):
                _ = model(dummy)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    mem_mb = torch.cuda.max_memory_allocated() / 1024**2
    return {'mean_ms': np.mean(times), 'std_ms': np.std(times), 'gpu_mem_mb': mem_mb}


print("Metrics functions ready")

# %% [markdown]
# ## Cell 3: Reproducibility Helpers & Training Function

# %%
def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_dataloader(dataset, shuffle, seed):
    g = torch.Generator()
    g.manual_seed(seed)
    def seed_worker(worker_id):
        ws = torch.initial_seed() % 2**32
        np.random.seed(ws); random.seed(ws)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=0, worker_init_fn=seed_worker, generator=g)


def train_single(model, name, epochs=EPOCHS, lr=3e-4, seed=42, warmup_epochs=0):
    """Train a single model with a given seed. Returns best metrics dict."""
    set_seed(seed)
    print(f"\n--- {name} | seed={seed} | lr={lr} ---")

    # NOTE on Batch Normalization:
    # BN layers in the frozen encoder remain in train mode during model.train(),
    # allowing running statistics to adapt to the Sentinel-2 domain.
    # This is intentional — frozen BN (eval mode) would prevent domain adaptation.

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    scaler = GradScaler()

    train_dl = make_dataloader(train_ds, shuffle=True, seed=seed)
    val_dl = make_dataloader(val_ds, shuffle=False, seed=seed)

    bests = {k: {'value': 0.0, 'metrics': {}}
             for k in ['miou', 'f1', 'accuracy', 'debris']}
    start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, masks in train_dl:
            if imgs.dim() == 4 and imgs.shape[-1] == 11:
                imgs = imgs.permute(0, 3, 1, 2)
            imgs = F.interpolate(imgs, size=(IMG_SIZE, IMG_SIZE),
                                 mode='bilinear', align_corners=False)
            imgs, masks = imgs.float().to(device), masks.long().to(device)
            optimizer.zero_grad()
            with autocast("cuda"):
                logits = model(imgs)
                if logits.shape[2:] != masks.shape[1:]:
                    logits = F.interpolate(logits, masks.shape[1:], mode='bilinear')
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        # Linear warmup then cosine decay
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / max(1, warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr
        else:
            scheduler.step()

        # Validation
        model.eval()
        val_m = {'mIoU': 0, 'F1': 0, 'Accuracy': 0, 'Debris_IoU': 0}
        with torch.no_grad():
            for imgs, masks in val_dl:
                if imgs.dim() == 4 and imgs.shape[-1] == 11:
                    imgs = imgs.permute(0, 3, 1, 2)
                imgs = F.interpolate(imgs, size=(IMG_SIZE, IMG_SIZE),
                                     mode='bilinear', align_corners=False)
                imgs, masks = imgs.float().to(device), masks.long().to(device)
                with autocast("cuda"):
                    logits = model(imgs)
                    if logits.shape[2:] != masks.shape[1:]:
                        logits = F.interpolate(logits, masks.shape[1:], mode='bilinear')
                met = compute_metrics(logits, masks)
                for k in val_m: val_m[k] += met[k]
        for k in val_m: val_m[k] /= len(val_dl)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Ep {epoch+1:02d} | Loss={total_loss/len(train_dl):.4f} | "
                  f"mIoU={val_m['mIoU']:.4f} | F1={val_m['F1']:.4f} | "
                  f"DebrisIoU={val_m['Debris_IoU']:.4f}")

        keys_map = {'miou': 'mIoU', 'f1': 'F1', 'accuracy': 'Accuracy', 'debris': 'Debris_IoU'}
        for bk, mk in keys_map.items():
            if val_m[mk] > bests[bk]['value']:
                bests[bk]['value'] = val_m[mk]
                bests[bk]['metrics'] = val_m.copy()
                # To prevent Kaggle 'No space left on device', ONLY save the checkpoint for best mIoU
                if bk == 'miou':
                    torch.save(model.state_dict(), f"{SAVE_DIR}/{name}_seed{seed}_best_miou.pth")

    bests['wall_time_min'] = (time.time() - start) / 60
    print(f"  Done in {bests['wall_time_min']:.1f} min | "
          f"Best mIoU={bests['miou']['value']:.4f}")
    return bests


def train_multi_seed(model_fn, name, epochs=EPOCHS, lr=3e-4, seeds=SEEDS, warmup_epochs=0):
    """
    Train across multiple seeds. model_fn() must return a fresh model each time.
    Returns: dict with mean, std, and per-seed results.
    """
    all_runs = []
    for seed in seeds:
        model = model_fn().to(device)
        result = train_single(model, name, epochs=epochs, lr=lr, seed=seed, warmup_epochs=warmup_epochs)
        all_runs.append(result)
        del model
        torch.cuda.empty_cache()

    # Aggregate
    metrics_keys = ['miou', 'f1', 'accuracy', 'debris']
    agg = {}
    for mk in metrics_keys:
        vals = [r[mk]['value'] for r in all_runs]
        agg[mk] = {'mean': np.mean(vals), 'std': np.std(vals), 'values': vals}
    agg['wall_time_min'] = np.mean([r['wall_time_min'] for r in all_runs])
    agg['name'] = name
    agg['runs'] = all_runs

    # Save aggregated results
    save_path = f"{SAVE_DIR}/{name}_results.json"
    serializable = {mk: {'mean': float(agg[mk]['mean']), 'std': float(agg[mk]['std'])}
                    for mk in metrics_keys}
    serializable['wall_time_min'] = float(agg['wall_time_min'])
    with open(save_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULT: {name}")
    for mk in metrics_keys:
        print(f"  {mk}: {agg[mk]['mean']:.4f} ± {agg[mk]['std']:.4f}")
    print(f"  Avg wall time: {agg['wall_time_min']:.1f} min")
    print(f"{'='*60}")
    return agg


print("Training infrastructure ready")

# %% [markdown]
# ## Cell 4: Model Creation Functions

# %%
def create_unet_baseline():
    """U-Net with ResNet50 encoder (all params trainable)."""
    return Unet(encoder_name="resnet50", encoder_weights="imagenet",
                in_channels=11, classes=NUM_CLASSES)


def create_unet_lora(lora_r=8,
                     target_modules=None,
                     decoder_mode='full'):
    """
    Create U-Net with LoRA on encoder conv layers.

    Args:
        lora_r: LoRA rank
        target_modules: which encoder modules to apply LoRA to
        decoder_mode: 'full' (all trainable), 'frozen', or 'lora'

    Design justification for Conv-LoRA:
      LoRA on convolutional layers factorizes weight updates as low-rank
      perturbations, analogous to attention projection adaptation in
      Transformers. See: FacT (ICLR 2024), Conv-LoRA (Yeh et al., 2024).
    """
    if target_modules is None:
        target_modules = ["conv1", "conv2", "conv3", "conv4", "downsample.0"]

    model = create_unet_baseline()

    # --- Encoder LoRA ---
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["decoder", "segmentation_head"] if decoder_mode == 'full' else []
    )
    model.encoder = get_peft_model(model.encoder, lora_cfg)

    # --- Decoder handling ---
    if decoder_mode == 'frozen':
        for p in model.decoder.parameters():
            p.requires_grad = False
        for p in model.segmentation_head.parameters():
            p.requires_grad = False
    elif decoder_mode == 'lora':
        # Apply LoRA to decoder conv layers too
        # SMP U-Net decoder blocks use conv1/conv2 inside DoubleConv sub-blocks
        import re
        dec_conv_names = [n for n, m in model.decoder.named_modules()
                         if isinstance(m, nn.Conv2d) and m.kernel_size[0] > 1]
        # Use full dotted paths — SMP wraps Conv2d inside Conv2dReLU Sequential,
        # so the last segment alone (e.g. "0") is ambiguous.  Full paths like
        # "blocks.0.conv1.0" uniquely identify each raw Conv2d layer.
        dec_targets = dec_conv_names  # full paths are what PEFT needs
        if dec_targets:
            dec_lora_cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_r * 2,
                target_modules=dec_targets,
                lora_dropout=0.1, bias="none"
            )
            model.decoder = get_peft_model(model.decoder, dec_lora_cfg)
        else:
            print(f"  WARNING: No conv targets found in decoder, falling back to frozen decoder")
            for p in model.decoder.parameters():
                p.requires_grad = False
    # decoder_mode == 'full': decoder stays fully trainable (default)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc_total = sum(p.numel() for p in model.encoder.parameters())
    enc_train = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    print(f"  LoRA r={lora_r} | decoder={decoder_mode} | "
          f"trainable: {trainable:,}/{total:,} ({100*trainable/total:.1f}%) | "
          f"encoder: {enc_train:,}/{enc_total:,} ({100*enc_train/enc_total:.1f}%)")
    return model


print("Model factories ready")

# %% [markdown]
# ## Cell 5: Experiment 1 — Learning Rate Ablation
# **Critical fix**: Compare baseline and LoRA at the SAME learning rates to eliminate confound.

# %%
print("="*80)
print("EXPERIMENT 1: LEARNING RATE ABLATION")
print("="*80)

LR_VALUES = [5e-5, 1e-4, 3e-4]
lr_results = {}

# Baseline at each LR
for lr_val in LR_VALUES:
    tag = f"baseline_lr{lr_val}"
    lr_results[tag] = train_multi_seed(
        model_fn=create_unet_baseline,
        name=tag, lr=lr_val, epochs=ABLATION_EPOCHS)

# LoRA r=8 at each LR
for lr_val in LR_VALUES:
    tag = f"lora_r8_lr{lr_val}"
    lr_results[tag] = train_multi_seed(
        model_fn=lambda: create_unet_lora(lora_r=8, decoder_mode='full'),
        name=tag, lr=lr_val, epochs=ABLATION_EPOCHS)

# Print LR ablation table
print("\n" + "="*80)
print("TABLE: LR ABLATION (mean ± std over 3 seeds)")
print("="*80)
rows = []
for key, res in lr_results.items():
    rows.append({
        'Method': key,
        'mIoU': f"{res['miou']['mean']:.4f}±{res['miou']['std']:.4f}",
        'F1': f"{res['f1']['mean']:.4f}±{res['f1']['std']:.4f}",
        'Debris IoU': f"{res['debris']['mean']:.4f}±{res['debris']['std']:.4f}",
        'Time (min)': f"{res['wall_time_min']:.1f}"
    })
print(pd.DataFrame(rows).to_markdown(index=False))

# Determine best LR for each method
best_baseline_lr = max(
    [(lr, lr_results[f"baseline_lr{lr}"]['miou']['mean']) for lr in LR_VALUES],
    key=lambda x: x[1])[0]
best_lora_lr = max(
    [(lr, lr_results[f"lora_r8_lr{lr}"]['miou']['mean']) for lr in LR_VALUES],
    key=lambda x: x[1])[0]
print(f"\nBest baseline LR: {best_baseline_lr}")
print(f"Best LoRA LR: {best_lora_lr}")

# %% [markdown]
# ## Cell 6: Experiment 2 — Rank Ablation (at best LR)

# %%
print("="*80)
print(f"EXPERIMENT 2: RANK ABLATION (lr={best_lora_lr})")
print("="*80)

rank_results = {}
for r in [2, 4, 16, 32]:  # 8 is reused below
    tag = f"lora_r{r}_rankablation"
    rank_results[tag] = train_multi_seed(
        model_fn=lambda r=r: create_unet_lora(lora_r=r, decoder_mode='full'),
        name=tag, lr=best_lora_lr, epochs=ABLATION_EPOCHS)

# Skip redundant training for r=8, reuse result from Exp 1 (saves ~1.25 hours)
tag_r8 = "lora_r8_rankablation"
rank_results[tag_r8] = lr_results[f"lora_r8_lr{best_lora_lr}"]

# Include best baseline for comparison
rank_results['baseline_best'] = lr_results[f"baseline_lr{best_baseline_lr}"]

print("\n" + "="*80)
print("TABLE: RANK ABLATION (mean ± std)")
print("="*80)
rows = []
for key, res in rank_results.items():
    rows.append({
        'Method': key,
        'mIoU': f"{res['miou']['mean']:.4f}±{res['miou']['std']:.4f}",
        'F1': f"{res['f1']['mean']:.4f}±{res['f1']['std']:.4f}",
        'Debris IoU': f"{res['debris']['mean']:.4f}±{res['debris']['std']:.4f}",
    })
print(pd.DataFrame(rows).to_markdown(index=False))

# Find best rank
best_rank = max(
    [(r, rank_results[f"lora_r{r}_rankablation"]['miou']['mean']) for r in [2, 4, 8, 16, 32]],
    key=lambda x: x[1])[0]
print(f"\nBest rank: r={best_rank}")

# %% [markdown]
# ## Cell 7: Experiment 3 — Decoder Trainability Ablation

# %%
print("="*80)
print(f"EXPERIMENT 3: DECODER ABLATION (r={best_rank}, lr={best_lora_lr})")
print("="*80)

dec_results = {}
for dec_mode in ['frozen', 'lora']:  # 'full' is reused below
    tag = f"lora_r{best_rank}_dec_{dec_mode}"
    dec_results[tag] = train_multi_seed(
        model_fn=lambda dm=dec_mode: create_unet_lora(
            lora_r=best_rank, decoder_mode=dm),
        name=tag, lr=best_lora_lr, epochs=ABLATION_EPOCHS)

# Skip redundant training for 'full', reuse result from Exp 2 if available; else compute it.
if 'rank_results' in globals() and f"lora_r{best_rank}_rankablation" in rank_results:
    dec_results[f"lora_r{best_rank}_dec_full"] = rank_results[f"lora_r{best_rank}_rankablation"]
else:
    dec_results[f"lora_r{best_rank}_dec_full"] = train_multi_seed(
        model_fn=lambda: create_unet_lora(lora_r=best_rank, decoder_mode='full'),
        name=f"lora_r{best_rank}_dec_full", lr=best_lora_lr, epochs=ABLATION_EPOCHS)

print("\nTABLE: DECODER ABLATION (mean ± std)")
rows = []
for key, res in dec_results.items():
    rows.append({
        'Method': key,
        'mIoU': f"{res['miou']['mean']:.4f}±{res['miou']['std']:.4f}",
        'F1': f"{res['f1']['mean']:.4f}±{res['f1']['std']:.4f}",
        'Debris IoU': f"{res['debris']['mean']:.4f}±{res['debris']['std']:.4f}",
    })
print(pd.DataFrame(rows).to_markdown(index=False))

# %% [markdown]
# ## Cell 8: Experiment 4 — Target Module Ablation

# %%
print("="*80)
print(f"EXPERIMENT 4: TARGET MODULE ABLATION (r={best_rank}, lr={best_lora_lr})")
print("="*80)

MODULE_SETS = {
    'minimal': ["conv1", "conv2"],
    'standard': ["conv1", "conv2", "conv3", "conv4", "downsample.0"],
}

mod_results = {}
for label, modules in MODULE_SETS.items():
    tag = f"lora_r{best_rank}_mod_{label}"
    if label == 'standard':
        # Skip redundant training, 'standard' is the default used in Exp 2
        if 'rank_results' in globals() and f"lora_r{best_rank}_rankablation" in rank_results:
            mod_results[tag] = rank_results[f"lora_r{best_rank}_rankablation"]
        elif 'dec_results' in globals() and f"lora_r{best_rank}_dec_full" in dec_results:
            mod_results[tag] = dec_results[f"lora_r{best_rank}_dec_full"]
        else:
            mod_results[tag] = train_multi_seed(
                model_fn=lambda m=modules: create_unet_lora(
                    lora_r=best_rank, target_modules=m, decoder_mode='full'),
                name=tag, lr=best_lora_lr, epochs=ABLATION_EPOCHS)
    else:
        mod_results[tag] = train_multi_seed(
            model_fn=lambda m=modules: create_unet_lora(
                lora_r=best_rank, target_modules=m, decoder_mode='full'),
            name=tag, lr=best_lora_lr, epochs=ABLATION_EPOCHS)

print("\nTABLE: TARGET MODULE ABLATION (mean ± std)")
rows = []
for key, res in mod_results.items():
    rows.append({
        'Method': key,
        'mIoU': f"{res['miou']['mean']:.4f}±{res['miou']['std']:.4f}",
        'F1': f"{res['f1']['mean']:.4f}±{res['f1']['std']:.4f}",
        'Debris IoU': f"{res['debris']['mean']:.4f}±{res['debris']['std']:.4f}",
    })
print(pd.DataFrame(rows).to_markdown(index=False))

# %% [markdown]
# ## Cell 9: Experiment 5 — Architecture Generalization
# **Goal:** To robustly address assumptions about model architecture by proving LoRA's effectiveness across multiple standard segmentation architectures.

# %%
print("="*80)
print(f"EXPERIMENT 5: ARCHITECTURE GENERALIZATION (r={best_rank}, lr={best_lora_lr})")
print("="*80)

def create_fpn_baseline():
    return FPN(encoder_name="resnet50", encoder_weights="imagenet", in_channels=11, classes=NUM_CLASSES)

def create_fpn_lora():
    model = create_fpn_baseline()
    lora_cfg = LoraConfig(r=best_rank, lora_alpha=best_rank*2, target_modules=["conv1", "conv2", "conv3", "conv4", "downsample.0"], lora_dropout=0.1, bias="none", modules_to_save=["decoder", "segmentation_head"])
    model.encoder = get_peft_model(model.encoder, lora_cfg)
    return model

def create_upp_baseline():
    return UnetPlusPlus(encoder_name="resnet50", encoder_weights="imagenet", in_channels=11, classes=NUM_CLASSES)

def create_upp_lora():
    model = create_upp_baseline()
    lora_cfg = LoraConfig(r=best_rank, lora_alpha=best_rank*2, target_modules=["conv1", "conv2", "conv3", "conv4", "downsample.0"], lora_dropout=0.1, bias="none", modules_to_save=["decoder", "segmentation_head"])
    model.encoder = get_peft_model(model.encoder, lora_cfg)
    return model

arch_configs = {
    'FPN_Baseline': (create_fpn_baseline, best_baseline_lr),
    'FPN_LoRA': (create_fpn_lora, best_lora_lr),
    'UnetPlusPlus_Baseline': (create_upp_baseline, best_baseline_lr),
    'UnetPlusPlus_LoRA': (create_upp_lora, best_lora_lr),
}

arch_results = {}
for name, (fn, lr) in arch_configs.items():
    arch_results[name] = train_multi_seed(model_fn=fn, name=name, lr=lr, warmup_epochs=2)

print("\nTABLE: ARCHITECTURE GENERALIZATION (mean ± std)")
rows = []
for key, res in arch_results.items():
    rows.append({
        'Method': key,
        'mIoU': f"{res['miou']['mean']:.4f}±{res['miou']['std']:.4f}",
        'F1': f"{res['f1']['mean']:.4f}±{res['f1']['std']:.4f}",
        'Debris IoU': f"{res['debris']['mean']:.4f}±{res['debris']['std']:.4f}",
    })
print(pd.DataFrame(rows).to_markdown(index=False))

# %% [markdown]
# ## Cell 10: Experiment 6 — Data Efficiency
# **Goal:** Show that LoRA degrades more gracefully than full fine-tuning under limited labeled data.
# This is a critical finding for marine debris detection where labels are scarce.

# %%
print("="*80)
print(f"EXPERIMENT 6: DATA EFFICIENCY (r={best_rank}, lr baseline={best_baseline_lr}, lr lora={best_lora_lr})")
print("="*80)

from torch.utils.data import Subset

DATA_FRACTIONS = [0.25, 0.50, 0.75, 1.0]

def make_subset(dataset, fraction, seed=42):
    """Return a deterministic subset of the dataset."""
    n = len(dataset)
    k = max(1, int(n * fraction))
    rng = np.random.RandomState(seed)
    indices = rng.choice(n, k, replace=False).tolist()
    return Subset(dataset, indices)

data_eff_results = {}
_full_train_ds = train_ds

for frac in DATA_FRACTIONS:
    pct = int(frac * 100)
    if frac < 1.0:
        sub = make_subset(_full_train_ds, frac, seed=0)
        # Monkey-patch train_ds temporarily
        train_ds = sub
    else:
        train_ds = _full_train_ds

    n_samples = len(train_ds)
    print(f"\n{'='*60}")
    print(f"DATA FRACTION: {pct}% ({n_samples} samples)")
    print(f"{'='*60}")

    # Baseline at this fraction
    tag_bl = f"baseline_{pct}pct"
    data_eff_results[tag_bl] = train_multi_seed(
        model_fn=create_unet_baseline, name=tag_bl, lr=best_baseline_lr, warmup_epochs=2)

    # LoRA at this fraction
    tag_lo = f"lora_r{best_rank}_{pct}pct"
    data_eff_results[tag_lo] = train_multi_seed(
        model_fn=lambda: create_unet_lora(lora_r=best_rank, decoder_mode='full'),
        name=tag_lo, lr=best_lora_lr, warmup_epochs=2)

# Restore original train_ds
train_ds = _full_train_ds

print("\n" + "="*80)
print("TABLE: DATA EFFICIENCY (mean ± std)")
print("="*80)
rows = []
for frac in DATA_FRACTIONS:
    pct = int(frac * 100)
    bl = data_eff_results[f"baseline_{pct}pct"]
    lo = data_eff_results[f"lora_r{best_rank}_{pct}pct"]
    rows.append({
        'Data %': f"{pct}%",
        'Baseline mIoU': f"{bl['miou']['mean']:.4f}±{bl['miou']['std']:.4f}",
        'LoRA mIoU': f"{lo['miou']['mean']:.4f}±{lo['miou']['std']:.4f}",
        'Baseline F1': f"{bl['f1']['mean']:.4f}±{bl['f1']['std']:.4f}",
        'LoRA F1': f"{lo['f1']['mean']:.4f}±{lo['f1']['std']:.4f}",
        'Δ mIoU': f"{lo['miou']['mean'] - bl['miou']['mean']:+.4f}",
    })
print(pd.DataFrame(rows).to_markdown(index=False))
