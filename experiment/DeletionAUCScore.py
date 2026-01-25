import argparse
import csv
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision.transforms.functional import gaussian_blur
from tqdm import tqdm

# Ensure project root is on sys.path for relative imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset.utils import ImageFilelist, create_annotation_file, get_transformation
from experiment.build_model import get_model
import experiment.visualize_run as visualize_run
from torchvision.datasets.folder import ImageFolder


# Thin wrapper to ensure we always have a SimpleNamespace with expected fields.
def _load_params(cfg_path, data_path, top_traits):
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    params = SimpleNamespace(**cfg)
    # Ensure essentials for evaluation
    params.class_num = getattr(params, "class_num", 200)  # CUB has 200 classes
    params.vis_attn = getattr(params, "vis_attn", True)
    params.data_path = data_path
    params.top_traits = top_traits
    params.output_dir = getattr(params, "output_dir", "visualization/deletion_auc")
    params.vis_outdir = getattr(params, "vis_outdir", "visualization")
    params.test_batch_size = 1
    params.batch_size = getattr(params, "batch_size", 1)
    params.final_run = True
    params.debug = getattr(params, "debug", False)
    params.data = getattr(params, "data", "cub")
    params.model = getattr(params, "model", "dino")
    params.vpt_num = getattr(params, "vpt_num", 0)
    return params


def _prepare_dataloader(data_root, batch_size):
    # Use ImageFolder to automatically load all images from directory structure
    transform = get_transformation("val")
    dataset = ImageFolder(data_root, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


def _capture_pruned_attention(model, inputs, class_idx, params):
    captured = {}

    def _capture_overlay(samples, patch_size, attn_map, path):
        captured["attn_map"] = attn_map.detach()

    original_overlay = visualize_run.create_overlay_images
    original_combine = visualize_run.combine_images
    try:
        visualize_run.create_overlay_images = _capture_overlay
        visualize_run.combine_images = lambda *args, **kwargs: None
        visualize_run.prune_attn_heads(
            model,
            inputs,
            torch.tensor([class_idx], device=inputs.device),
            class_idx,
            0,
            params,
        )
    finally:
        visualize_run.create_overlay_images = original_overlay
        visualize_run.combine_images = original_combine
    return captured.get("attn_map")


def _normalize_heatmap(attn_map, input_spatial):
    # attn_map: (1, heads, tokens)
    if attn_map is None:
        return None
    attn_map = attn_map.mean(dim=1, keepdim=False)  # (1, tokens)
    num_tokens = attn_map.shape[-1]
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        return None
    attn_map = attn_map.view(1, 1, side, side)
    attn_map = F.interpolate(attn_map, size=input_spatial, mode="bilinear", align_corners=False)
    attn_map = attn_map.squeeze(0).squeeze(0)
    attn_min, attn_max = attn_map.min(), attn_map.max()
    heatmap = (attn_map - attn_min) / (attn_max - attn_min + 1e-6)
    return heatmap


def _deletion_curve(model, original, blurred, heatmap, target_idx, steps):
    """
    Deletion curve: start with original image, progressively blur the most salient pixels.
    Lower AUC is better - indicates salient pixels were critical for prediction.
    """
    device = original.device
    c, h, w = original.shape[1:]
    flat_heatmap = heatmap.view(-1)
    # Sort in DESCENDING order - remove most salient pixels first
    sorted_indices = torch.argsort(flat_heatmap, descending=True)
    num_pixels = h * w
    fractions = np.linspace(0.0, 1.0, steps + 1)
    probs = []

    with torch.no_grad():
        for frac in fractions:
            k = max(1, int(frac * num_pixels)) if frac > 0 else 0
            if k == 0:
                # Step 0: original image (no pixels deleted)
                x_k = original
            else:
                # Create mask: 1 for pixels to DELETE (replace with blur)
                mask = torch.zeros(num_pixels, device=device)
                mask[sorted_indices[:k]] = 1.0
                mask = mask.view(1, 1, h, w)
                # Blend: keep original where mask=0, use blur where mask=1
                x_k = original * (1.0 - mask) + blurred * mask
            
            logits, _ = model(x_k)
            # Extract probability of ground-truth class only
            logits_flat = logits.view(-1)  # flatten to 1D
            if logits_flat.numel() == 1:
                # Single logit: use sigmoid as confidence for ground-truth
                prob = torch.sigmoid(logits_flat[0]).item()
            else:
                # Per-class logits: softmax and extract ground-truth class
                prob = torch.softmax(logits_flat, dim=-1)[int(target_idx)].item()
            probs.append(prob)
    return fractions, np.array(probs)


def compute_deletion_auc(args):
    params = _load_params(args.config, args.data_path, args.top_traits)

    model, _, _ = get_model(params)
    state = torch.load(args.checkpoint, map_location=params.device)
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()

    loader = _prepare_dataloader(args.data_path, batch_size=1)

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    per_image_scores = []

    # Prepare CSV for incremental logging
    csv_exists = os.path.isfile(args.output_csv)
    csv_file = open(args.output_csv, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(["image", "auc"])  # header

    for idx, (images, targets) in enumerate(tqdm(loader, desc="Deletion AUC")):
        images = images.to(params.device)
        targets = targets.to(params.device)
        target_cls = targets.item()
        image_rel_path = loader.dataset.imlist[idx][0] if hasattr(loader.dataset, "imlist") else str(idx)

        # Use raw attention without pruning
        _, raw_attn = model(images)
        attn_map = raw_attn[:, :, target_cls, (params.vpt_num + 1) :]

        heatmap = _normalize_heatmap(attn_map, input_spatial=images.shape[-2:])
        if heatmap is None:
            continue

        blurred = gaussian_blur(images, kernel_size=11, sigma=5.0)
        fractions, probs = _deletion_curve(model, images, blurred, heatmap, target_cls, args.steps)
        auc = np.trapz(probs, fractions)
        per_image_scores.append((image_rel_path, auc))

        # Incremental write for durability
        csv_writer.writerow([image_rel_path, auc])
        csv_file.flush()

        # Save plot if plot_dir specified
        if args.plot_dir:
            try:
                import matplotlib.pyplot as plt
                os.makedirs(args.plot_dir, exist_ok=True)
                plt.figure(figsize=(5, 4))
                plt.plot(fractions, probs, marker='o', linewidth=2, markersize=4, color='red')
                plt.fill_between(fractions, probs, alpha=0.2, color='red')
                plt.xlabel('Fraction of Pixels Deleted')
                plt.ylabel('P(ground-truth class)')
                plt.title(f'Deletion Curve (AUC={auc:.3f})\n{os.path.basename(image_rel_path)}')
                plt.grid(True, alpha=0.3)
                plt.ylim([0, 1.05])
                # Create safe filename from image path
                safe_name = image_rel_path.replace('\\', '_').replace('/', '_')
                plot_path = os.path.join(args.plot_dir, f"{safe_name}.png")
                os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                plt.savefig(plot_path, bbox_inches='tight', dpi=100)
                plt.close()
            except Exception as e:
                print(f"Plotting failed for sample {idx}: {e}")

    if len(per_image_scores) == 0:
        print("No samples processed; check data path or heatmap generation.")
        return

    mean_auc = float(np.mean([score for _, score in per_image_scores]))
    print(f"Mean Deletion AUC: {mean_auc:.4f} (lower is better)")

    # Close the CSV file (it already contains all rows incrementally)
    csv_file.close()
    print(f"Saved per-image scores to {args.output_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Deletion AUC Score for CUB")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config used to build the model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to fine-tuned checkpoint")
    parser.add_argument("--data-path", type=str, required=True, help="Root of the CUB dataset (with train/val folders)")
    parser.add_argument("--output-csv", type=str, default="output/deletion_auc_scores.csv", help="Where to save per-image AUC values")
    parser.add_argument("--steps", type=int, default=100, help="Number of deletion steps")
    parser.add_argument("--top-traits", type=int, default=4, help="Number of heads retained by prune_attention_heads")
    parser.add_argument("--plot-dir", type=str, default="", help="Directory to save per-image deletion curves (PNG)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compute_deletion_auc(args)
