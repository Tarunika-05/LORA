# %% [markdown]
# ## Cell 10: Test Set Evaluation
# Evaluate best models on the held-out test set (359 patches).
# **Run this ONCE after all experiments are finalized.**

# %%
print("="*80)
print("TEST SET EVALUATION")
print("="*80)

def evaluate_on_test(model, name):
    """Evaluate a model on the test set. Returns metrics + per-class IoU."""
    model.eval()
    test_dl = make_dataloader(test_ds, shuffle=False, seed=42)

    # Batch-by-batch metrics using raw logits (avoids double-argmax bug)
    val_m = {'mIoU': 0, 'F1': 0, 'Accuracy': 0, 'Debris_IoU': 0}
    all_preds_np, all_labels_np = [], []

    with torch.no_grad():
        for imgs, masks in test_dl:
            if imgs.dim() == 4 and imgs.shape[-1] == 11:
                imgs = imgs.permute(0, 3, 1, 2)
            imgs = F.interpolate(imgs, size=(IMG_SIZE, IMG_SIZE),
                                 mode='bilinear', align_corners=False)
            imgs, masks = imgs.float().to(device), masks.long().to(device)
            with autocast("cuda"):
                logits = model(imgs)
                if logits.shape[2:] != masks.shape[1:]:
                    logits = F.interpolate(logits, masks.shape[1:], mode='bilinear')
            # Pass raw logits (4D, 11 channels) — compute_metrics argmaxes correctly
            met = compute_metrics(logits, masks)
            for k in val_m: val_m[k] += met[k]
            # Accumulate for per-class IoU
            all_preds_np.append(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels_np.append(masks.cpu().numpy())

    for k in val_m: val_m[k] /= len(test_dl)

    # Per-class IoU on full test set
    all_preds_np = np.concatenate(all_preds_np)
    all_labels_np = np.concatenate(all_labels_np)
    valid = all_labels_np != -1
    preds_v, labels_v = all_preds_np[valid], all_labels_np[valid]
    per_class = {}
    for cls in range(NUM_CLASSES):
        inter = ((preds_v == cls) & (labels_v == cls)).sum()
        union = ((preds_v == cls) | (labels_v == cls)).sum()
        per_class[CLASS_NAMES[cls]] = float(inter / union) if union > 0 else float('nan')

    return val_m, per_class


# Load and evaluate best models
test_configs = {
    'Baseline (best LR)': (create_unet_baseline,
                           f"{SAVE_DIR}/baseline_lr{best_baseline_lr}_seed42_best_miou.pth"),
    f'LoRA r={best_rank} (best LR)': (
        lambda: create_unet_lora(lora_r=best_rank, decoder_mode='full'),
        f"{SAVE_DIR}/lora_r{best_rank}_rankablation_seed42_best_miou.pth"),
}

test_results = {}
per_class_results = {}
for name, (model_fn, ckpt_path) in test_configs.items():
    if os.path.exists(ckpt_path):
        model = model_fn().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        info = model.load_state_dict(ckpt, strict=False)
        if info.missing_keys:
            print(f"  Warning: {len(info.missing_keys)} missing keys during load")
        metrics, per_class = evaluate_on_test(model, name)
        test_results[name] = metrics
        per_class_results[name] = per_class
        print(f"\n{name}: mIoU={metrics['mIoU']:.4f} | F1={metrics['F1']:.4f} | "
              f"Debris IoU={metrics['Debris_IoU']:.4f}")
        del model; torch.cuda.empty_cache()

if not test_results:
    print("No checkpoints found from Notebook 1A — test-set evaluation skipped.")
    print("(This is expected when running Notebooks 1B/2 independently.)")
    print("Validation metrics reported above are the primary comparison.")

# Per-class IoU table
if per_class_results:
    print("\nPER-CLASS IoU ON TEST SET:")
    pc_df = pd.DataFrame(per_class_results).round(4)
    print(pc_df.to_markdown())

# %% [markdown]
# ## Cell 11: Inference Time & Memory Analysis

# %%
print("="*80)
print("INFERENCE TIME ANALYSIS")
print("="*80)

inf_configs = {
    'Baseline': create_unet_baseline,
    f'LoRA r={best_rank}': lambda: create_unet_lora(lora_r=best_rank, decoder_mode='full'),
}

inf_rows = []
for name, model_fn in inf_configs.items():
    model = model_fn().to(device).eval()
    torch.cuda.reset_peak_memory_stats()
    timing = measure_inference_time(model, num_runs=50)
    inf_rows.append({
        'Method': name,
        'Inference (ms)': f"{timing['mean_ms']:.2f} ± {timing['std_ms']:.2f}",
        'GPU Memory (MB)': f"{timing['gpu_mem_mb']:.1f}",
    })
    total_p = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    inf_rows[-1]['Total Params'] = f"{total_p:,}"
    inf_rows[-1]['Trainable Params'] = f"{train_p:,}"
    inf_rows[-1]['Trainable %'] = f"{100*train_p/total_p:.1f}%"
    del model; torch.cuda.empty_cache()

print(pd.DataFrame(inf_rows).to_markdown(index=False))

# --- EFFICIENCY SUMMARY TABLE ---
print("\n" + "="*80)
print("PRACTICAL DEPLOYMENT EFFICIENCY")
print("="*80)

eff_rows = []
for row in inf_rows:
    train_time_str = "N/A"
    if 'lr_results' in globals():
        for k, v in globals()['lr_results'].items():
            if 'baseline' in k and 'Baseline' in row['Method']:
                train_time_str = f"{v['wall_time_min']:.1f} min"
                break
            elif 'lora' in k and 'LoRA' in row['Method']:
                train_time_str = f"{v['wall_time_min']:.1f} min"
                break
    
    eff_rows.append({
        'Method': row['Method'],
        'Trainable Params': row['Trainable Params'],
        'Total Params': row['Total Params'],
        'Trainable %': row['Trainable %'],
        'GPU Mem (Val)': row['GPU Memory (MB)'] + " MB",
        'Train Time (70 ep)': train_time_str
    })

print(pd.DataFrame(eff_rows).to_markdown(index=False))

# %% [markdown]
# ## Cell 12: Qualitative Visualizations


# Color map for 11 classes
cmap_colors = plt.cm.tab20(np.linspace(0, 1, NUM_CLASSES))
seg_cmap = mcolors.ListedColormap(cmap_colors)

def visualize_predictions(models_dict, dataset, indices=None, n=5):
    """Side-by-side: RGB composite | Ground Truth | Model predictions."""
    if indices is None:
        indices = np.random.choice(len(dataset), n, replace=False)

    n_models = len(models_dict)
    fig, axes = plt.subplots(n, 2 + n_models, figsize=(4*(2+n_models), 4*n))

    for row_i, idx in enumerate(indices):
        img, mask = dataset[idx]
        if img.dim() == 2 or (img.dim() == 3 and img.shape[-1] == 11):
            if img.dim() == 3 and img.shape[-1] == 11:
                img_t = img.permute(2, 0, 1) if img.shape[-1] == 11 else img
            else:
                img_t = img
        else:
            img_t = img

        # RGB composite (bands 4,3,2 for Sentinel-2)
        if img_t.shape[0] >= 4:
            rgb = img_t[[3, 2, 1]].numpy()
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            rgb = np.clip(rgb.transpose(1, 2, 0), 0, 1)
        else:
            rgb = np.zeros((IMG_SIZE, IMG_SIZE, 3))

        ax = axes[row_i, 0] if n > 1 else axes[0]
        ax.imshow(rgb)
        ax.set_title('RGB Composite' if row_i == 0 else '')
        ax.axis('off')

        ax = axes[row_i, 1] if n > 1 else axes[1]
        ax.imshow(mask.numpy(), cmap=seg_cmap, vmin=0, vmax=NUM_CLASSES-1,
                  interpolation='nearest')
        ax.set_title('Ground Truth' if row_i == 0 else '')
        ax.axis('off')

        # Model predictions
        for col_i, (mname, model) in enumerate(models_dict.items()):
            model.eval()
            inp = img_t.unsqueeze(0).float().to(device)
            inp = F.interpolate(inp, size=(IMG_SIZE, IMG_SIZE),
                                mode='bilinear', align_corners=False)
            with torch.no_grad():
                with autocast("cuda"):
                    logits = model(inp)
            pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()

            ax = axes[row_i, 2+col_i] if n > 1 else axes[2+col_i]
            ax.imshow(pred, cmap=seg_cmap, vmin=0, vmax=NUM_CLASSES-1,
                      interpolation='nearest')
            ax.set_title(mname if row_i == 0 else '')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/qualitative_results.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to {SAVE_DIR}/qualitative_results.png")


# Load best baseline and best LoRA for visualization
viz_models = {}
for name, (model_fn, ckpt_path) in test_configs.items():
    if os.path.exists(ckpt_path):
        model = model_fn().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
        model.eval()
        viz_models[name] = model

if viz_models:
    print("="*80)
    print("QUALITATIVE RESULTS")
    print("="*80)
    visualize_predictions(viz_models, test_ds, n=5)
    # Clean up
    for m in viz_models.values(): del m
    torch.cuda.empty_cache()

    # Add legend
    fig, ax = plt.subplots(1, 1, figsize=(8, 2))
    for i, name in enumerate(CLASS_NAMES):
        ax.barh(0, 1, left=i, color=cmap_colors[i], label=name)
    ax.set_xlim(0, NUM_CLASSES)
    ax.axis('off')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=8)
    plt.title("Class Legend")
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/class_legend.png", dpi=150, bbox_inches='tight')
    plt.show()
# %% [markdown]
# ## Cell 13: Comprehensive Summary Tables

# %%
print("="*80)
print("COMPREHENSIVE SUMMARY")
print("="*80)

def fmt(agg, key):
    return f"{agg[key]['mean']:.4f} ± {agg[key]['std']:.4f}"

# Collect all experiment results that exist
all_exp = {}
for name in ['lr_results', 'rank_results', 'dec_results', 'mod_results', 'arch_results', 'data_eff_results']:
    if name in dir():
        all_exp.update(eval(name))

if all_exp:
    rows = []
    for key, res in all_exp.items():
        if 'miou' in res and isinstance(res['miou'], dict) and 'mean' in res['miou']:
            rows.append({
                'Experiment': key,
                'mIoU': fmt(res, 'miou'),
                'F1': fmt(res, 'f1'),
                'Debris IoU': fmt(res, 'debris'),
                'Accuracy': fmt(res, 'accuracy'),
            })
    if rows:
        print("\nALL EXPERIMENTS (mean ± std, 3 seeds):")
        print(pd.DataFrame(rows).to_markdown(index=False))

# Test set results
if test_results:
    print("\nTEST SET RESULTS:")
    test_rows = [{'Method': k, **{mk: f"{v:.4f}" for mk, v in m.items()}}
                 for k, m in test_results.items()]
    print(pd.DataFrame(test_rows).to_markdown(index=False))

print("\n" + "="*80)
print("ALL EXPERIMENTS COMPLETE")
print("="*80)
