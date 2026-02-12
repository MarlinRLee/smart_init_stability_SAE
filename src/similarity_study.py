"""
Similarity Study: Track dictionary similarity during SAE training.

Trains multiple SI-SAE models with controlled seeds and tracks:
1. Similarity between each model and its initialization (with noise)
2. Similarity between each model and the clean k-means centers (without noise)
3. Pairwise similarity between models with same init+noise (same seed)
4. Pairwise similarity between models with different noise (different seed)

Results are saved as JSON for plotting with plot_similarity.py.
"""

import os
import sys
import json
import time
import argparse
import glob as globmod
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np

from data import get_dataset_stats, GPUNormalizer, create_dataloader, DeviceDataLoader
from utils import cosine_kmeans
from overcomplete.sae import TopKSAE
from overcomplete.sae.train import extract_input
from overcomplete.metrics import l2, l0_eps
from overcomplete.sae.trackers import DeadCodeTracker
from train import EarlyStopping, validate
from common import GracefulKiller


# ── Similarity metrics ──────────────────────────────────────────────────────

def mean_max_cosine_similarity(D1, D2):
    """
    Fast similarity: for each atom in D1, find the max cosine sim to any atom in D2.
    Returns the average of these maxima. Symmetric version averages both directions.
    """
    D1_norm = D1 / (D1.norm(dim=1, keepdim=True) + 1e-6)
    D2_norm = D2 / (D2.norm(dim=1, keepdim=True) + 1e-6)
    sim = torch.matmul(D1_norm, D2_norm.T)  # (n1, n2)
    max_1to2 = sim.max(dim=1).values.mean().item()
    max_2to1 = sim.max(dim=0).values.mean().item()
    return (max_1to2 + max_2to1) / 2


def cosine_similarity_matrix(D1, D2):
    """Full cosine similarity matrix between two dictionaries."""
    D1_norm = D1 / (D1.norm(dim=1, keepdim=True) + 1e-6)
    D2_norm = D2 / (D2.norm(dim=1, keepdim=True) + 1e-6)
    return torch.matmul(D1_norm, D2_norm.T)


# ── SI-SAE creation with controlled seed ────────────────────────────────────

def create_si_sae(d_brain, d_model, k, device, centers, per_init, init_seed):
    """
    Create an SI-SAE with a specific random seed controlling the noise.

    Parameters
    ----------
    centers : torch.Tensor
        Clean k-means centers (d_model, d_brain).
    per_init : float
        Noise level for SI initialization.
    init_seed : int
        Random seed for the noise generation.

    Returns
    -------
    sae : TopKSAE
        The initialized model.
    init_dict : torch.Tensor
        The initial dictionary weights (for similarity tracking).
    """
    # Create base model (seed doesn't matter much here, we overwrite weights)
    sae = TopKSAE(input_shape=d_brain, nb_concepts=d_model, top_k=k, device=device)

    # Generate noise with controlled seed
    rng = torch.Generator()
    rng.manual_seed(init_seed)
    noise = torch.randn(centers.shape, generator=rng) * centers.std() * per_init
    weights = (1 - per_init) * centers + noise

    # Set dictionary and encoder
    norm_weights = torch.nn.functional.normalize(weights.to(device), p=2, dim=-1)
    sae.dictionary._weights.data = norm_weights
    enc_weights = norm_weights.clone() * (1.0 / (k ** 0.5))
    sae.encoder.final_block[0].weight.data = enc_weights

    init_dict = norm_weights.detach().clone().cpu()
    return sae, init_dict


# ── Training with similarity tracking ───────────────────────────────────────

def criterion(x, x_hat, pre_codes, codes, dictionary):
    """Reconstruction loss"""
    loss = (x - x_hat).square().mean()

    return loss


def train_with_similarity_tracking(
    model, dataloader, optimizer, scheduler,
    nb_epochs, device, model_name,
    init_dict, clean_centers,
    other_models=None,
    clip_grad=1.0,
    use_mixed_precision=False,
    freeze_dict_epochs=2,
    val_loader=None,
):
    """
    Train an SAE and track dictionary similarity at each epoch.

    Parameters
    ----------
    init_dict : torch.Tensor
        Initial dictionary weights (with noise) on CPU.
    clean_centers : torch.Tensor
        Clean k-means centers (without noise) on CPU.
    other_models : dict, optional
        {name: model} dict of other models to compute pairwise similarity with.

    Returns
    -------
    logs : dict
        Training logs including similarity trajectories.
    """
    logs = defaultdict(list)
    global_step = 0
    frozen = freeze_dict_epochs > 0

    if frozen:
        for param in model.dictionary.parameters():
            param.requires_grad = False

    # Move references to device for similarity computation
    init_dict_dev = init_dict.to(device)
    clean_centers_dev = clean_centers.to(device)

    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    for epoch in range(nb_epochs):
        if frozen and epoch >= freeze_dict_epochs:
            for param in model.dictionary.parameters():
                param.requires_grad = True
            print(f"  [{model_name}] Unfreezing dictionary at epoch {epoch+1}")
            frozen = False

        model.train()
        start_time = time.time()
        epoch_loss = 0.0
        batch_count = 0
        mon_count = 0
        dead_tracker = None

        for batch in dataloader:
            global_step += 1
            batch_count += 1
            x = extract_input(batch)
            optimizer.zero_grad(set_to_none=True)

            if use_mixed_precision:
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    z_pre, z, x_hat = model(x)
                loss = criterion(x.float(), x_hat.float(), z_pre.float(),
                                 z.float(), model.get_dictionary().float())
                if dead_tracker is None:
                    dead_tracker = DeadCodeTracker(z.shape[1], device)
                dead_tracker.update(z.float())
                loss.backward()
            else:
                x = x.float()
                z_pre, z, x_hat = model(x)
                loss = criterion(x, x_hat, z_pre, z, model.get_dictionary())
                if dead_tracker is None:
                    dead_tracker = DeadCodeTracker(z.shape[1], device)
                dead_tracker.update(z)
                loss.backward()

            if clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if batch_count % 50 == 0:
                mon_count += 1
                epoch_loss += loss.item()

        epoch_duration = time.time() - start_time

        # ── Compute similarities at end of epoch ──
        model.eval()
        with torch.no_grad():
            D = model.get_dictionary().detach()

            sim_to_init = mean_max_cosine_similarity(D, init_dict_dev)
            sim_to_clean = mean_max_cosine_similarity(D, clean_centers_dev)

            logs['sim_to_init'].append(sim_to_init)
            logs['sim_to_clean_centers'].append(sim_to_clean)

            # Pairwise similarities with other models
            if other_models:
                for other_name, other_model in other_models.items():
                    other_model.eval()
                    D_other = other_model.get_dictionary().detach()
                    pair_sim = mean_max_cosine_similarity(D, D_other)
                    logs[f'sim_to_{other_name}'].append(pair_sim)

        avg_loss = epoch_loss / mon_count if mon_count > 0 else float('nan')
        dead_ratio = dead_tracker.get_dead_ratio() if dead_tracker else 0.0
        logs['avg_loss'].append(avg_loss)
        logs['dead_features'].append(dead_ratio)
        logs['time_epoch'].append(epoch_duration)

        # Validation
        val_msg = ""
        if val_loader is not None:
            val_metrics = validate(model, val_loader, criterion, device)
            logs['val_loss'].append(val_metrics['val_loss'])
            val_msg = f" | Val loss: {val_metrics['val_loss']:.4f}"

        pair_str = ""
        if other_models:
            pair_strs = [f"{k}: {logs[k][-1]:.4f}" for k in logs if k.startswith('sim_to_') and k not in ('sim_to_init', 'sim_to_clean_centers')]
            if pair_strs:
                pair_str = " | " + ", ".join(pair_strs)

        print(f"  [{model_name}] Epoch {epoch+1:3d}/{nb_epochs} "
              f"Loss: {avg_loss:.4f} Dead: {dead_ratio*100:.1f}% "
              f"SimInit: {sim_to_init:.4f} SimClean: {sim_to_clean:.4f}"
              f"{pair_str}{val_msg} ({epoch_duration:.1f}s)")

    return dict(logs)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train SI-SAEs and track dictionary similarity over training"
    )
    parser.add_argument("shard_directory", type=str,
                        help="Directory containing shard_*.pt files")
    parser.add_argument("--centers-dir", type=str, default="../centers",
                        help="Directory for pre-computed centers files")
    parser.add_argument("--output-dir", type=str, default="../similarity_results",
                        help="Directory to save results")
    parser.add_argument("--epochs", type=int, default=40,
                        help="Number of training epochs")
    parser.add_argument("--d-model", type=int, default=5000,
                        help="Dictionary size (number of SAE concepts)")
    parser.add_argument("--k-fraction", type=float, default=0.01,
                        help="Sparsity fraction")
    parser.add_argument("--per-init", type=float, default=0.1,
                        help="SI noise level")
    parser.add_argument("--batch-size", type=int, default=16384,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--mixed-precision", action="store_true",
                        help="Use bfloat16 mixed precision (recommended for A100)")
    parser.add_argument("--val-dir", type=str, default=None,
                        help="Validation shard directory")
    parser.add_argument("--seed-a", type=int, default=42,
                        help="Init seed for models A and B (same noise)")
    parser.add_argument("--seed-c", type=int, default=43,
                        help="Init seed for model C (different noise)")
    parser.add_argument("--train-seed-a", type=int, default=100,
                        help="Training seed for model A")
    parser.add_argument("--train-seed-b", type=int, default=200,
                        help="Training seed for model B")
    parser.add_argument("--train-seed-c", type=int, default=300,
                        help="Training seed for model C")
    parser.add_argument("--dataset-size", type=int, default=None,
                        help="Override dataset size (for scheduler computation)")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Running on {torch.cuda.get_device_name(0)}, TF32 enabled")

    os.makedirs(args.output_dir, exist_ok=True)

    k = int(args.k_fraction * args.d_model)
    print(f"Config: d_model={args.d_model}, k={k}, per_init={args.per_init}, "
          f"epochs={args.epochs}, batch_size={args.batch_size}")

    # ── Data loading ──
    shard_files = sorted(globmod.glob(os.path.join(args.shard_directory, 'shard_*.pt')))
    if not shard_files:
        raise FileNotFoundError(f"No shard_*.pt files in {args.shard_directory}")
    print(f"Found {len(shard_files)} training shards")

    first_shard = torch.load(shard_files[0], map_location='cpu', weights_only=True)
    d_brain = first_shard.shape[-1]
    samples_per_shard = first_shard.shape[0]
    dataset_size = args.dataset_size or len(shard_files) * samples_per_shard
    print(f"d_brain={d_brain}, estimated dataset_size={dataset_size}")

    mean, std = get_dataset_stats(args.shard_directory)
    normalizer = GPUNormalizer(mean, std).to(device)
    raw_loader = create_dataloader(args.shard_directory, args.batch_size,
                                   num_workers=8, prefetch_factor=2)
    loader = DeviceDataLoader(raw_loader, device, normalizer)

    # Validation loader
    val_loader = None
    if args.val_dir:
        try:
            val_mean, val_std = get_dataset_stats(args.val_dir)
            val_normalizer = GPUNormalizer(val_mean, val_std).to(device)
        except FileNotFoundError:
            val_normalizer = normalizer
        from data import create_val_dataloader
        raw_val = create_val_dataloader(args.val_dir, args.batch_size, num_workers=4, prefetch_factor=2)
        val_loader = DeviceDataLoader(raw_val, device, val_normalizer)
        print(f"Validation enabled")

    # ── Load / compute clean k-means centers ──
    si_path = os.path.join(args.centers_dir, f"si_centers_{args.d_model}.pt")
    if os.path.exists(si_path):
        print(f"Loading cached centers from {si_path}")
        clean_centers = torch.load(si_path, map_location='cpu', weights_only=True)
    else:
        print("Computing k-means centers (this may take a while)...")
        clean_centers = cosine_kmeans(loader, args.d_model, d_brain)
        os.makedirs(args.centers_dir, exist_ok=True)
        torch.save(clean_centers, si_path)
    print(f"Clean centers shape: {clean_centers.shape}")

    # ── Helper: create optimizer + scheduler ──
    def make_optimizer_scheduler(model):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        num_step_epoch = dataset_size // args.batch_size
        total_steps = num_step_epoch * args.epochs
        warmup_steps = total_steps // 4
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_steps)
        decay = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(total_steps - warmup_steps), eta_min=args.lr * 0.05)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])
        return optimizer, scheduler

    killer = GracefulKiller()

    # ══════════════════════════════════════════════════════════════════════════
    # Create 3 models:
    #   A: seed_a noise, train_seed_a training
    #   B: seed_a noise, train_seed_b training  (same init as A)
    #   C: seed_c noise, train_seed_c training  (different noise from A/B)
    # ══════════════════════════════════════════════════════════════════════════

    print("\n── Creating Model A (seed_a noise) ──")
    sae_a, init_dict_a = create_si_sae(
        d_brain, args.d_model, k, device, clean_centers, args.per_init, args.seed_a)

    print("── Creating Model B (seed_a noise, same init as A) ──")
    sae_b, init_dict_b = create_si_sae(
        d_brain, args.d_model, k, device, clean_centers, args.per_init, args.seed_a)

    print("── Creating Model C (seed_c noise, different from A/B) ──")
    sae_c, init_dict_c = create_si_sae(
        d_brain, args.d_model, k, device, clean_centers, args.per_init, args.seed_c)

    # Verify A and B have same init, C is different
    ab_init_sim = mean_max_cosine_similarity(init_dict_a.to(device), init_dict_b.to(device))
    ac_init_sim = mean_max_cosine_similarity(init_dict_a.to(device), init_dict_c.to(device))
    print(f"\nInit similarity A↔B: {ab_init_sim:.6f} (should be ~1.0)")
    print(f"Init similarity A↔C: {ac_init_sim:.6f} (should be < 1.0)")

    # ══════════════════════════════════════════════════════════════════════════
    # Train Model A
    # ══════════════════════════════════════════════════════════════════════════
    torch.manual_seed(args.train_seed_a)
    torch.cuda.manual_seed_all(args.train_seed_a)
    opt_a, sched_a = make_optimizer_scheduler(sae_a)
    logs_a = train_with_similarity_tracking(
        sae_a, loader, opt_a, sched_a,
        nb_epochs=args.epochs, device=device, model_name="A",
        init_dict=init_dict_a, clean_centers=clean_centers,
        use_mixed_precision=args.mixed_precision,
        val_loader=val_loader,
    )
    if killer.kill_now:
        print("Interrupted after Model A. Saving partial results.")

    # ══════════════════════════════════════════════════════════════════════════
    # Train Model B (tracking pairwise sim to A)
    # ══════════════════════════════════════════════════════════════════════════
    if not killer.kill_now:
        torch.manual_seed(args.train_seed_b)
        torch.cuda.manual_seed_all(args.train_seed_b)
        opt_b, sched_b = make_optimizer_scheduler(sae_b)
        logs_b = train_with_similarity_tracking(
            sae_b, loader, opt_b, sched_b,
            nb_epochs=args.epochs, device=device, model_name="B",
            init_dict=init_dict_b, clean_centers=clean_centers,
            other_models={"A": sae_a},
            use_mixed_precision=args.mixed_precision,
            val_loader=val_loader,
        )
    else:
        logs_b = {}

    # ══════════════════════════════════════════════════════════════════════════
    # Train Model C (tracking pairwise sim to A and B)
    # ══════════════════════════════════════════════════════════════════════════
    if not killer.kill_now:
        torch.manual_seed(args.train_seed_c)
        torch.cuda.manual_seed_all(args.train_seed_c)
        opt_c, sched_c = make_optimizer_scheduler(sae_c)
        logs_c = train_with_similarity_tracking(
            sae_c, loader, opt_c, sched_c,
            nb_epochs=args.epochs, device=device, model_name="C",
            init_dict=init_dict_c, clean_centers=clean_centers,
            other_models={"A": sae_a, "B": sae_b},
            use_mixed_precision=args.mixed_precision,
            val_loader=val_loader,
        )
    else:
        logs_c = {}

    # ══════════════════════════════════════════════════════════════════════════
    # Also retroactively compute A's similarity to B and C over B/C's epochs
    # Since A is already trained (frozen), we just need A's final dict vs B/C at each epoch.
    # But we already tracked B→A and C→A during their training.
    # ══════════════════════════════════════════════════════════════════════════

    # ── Save results ──
    results = {
        "config": {
            "d_model": args.d_model,
            "k": k,
            "k_fraction": args.k_fraction,
            "per_init": args.per_init,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed_a": args.seed_a,
            "seed_c": args.seed_c,
            "train_seed_a": args.train_seed_a,
            "train_seed_b": args.train_seed_b,
            "train_seed_c": args.train_seed_c,
            "shard_directory": args.shard_directory,
        },
        "init_similarities": {
            "A_B": ab_init_sim,
            "A_C": ac_init_sim,
        },
        "model_A": logs_a,
        "model_B": logs_b,
        "model_C": logs_c,
    }

    out_path = os.path.join(args.output_dir, "similarity_study_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Also save model state dicts
    for name, model in [("A", sae_a), ("B", sae_b), ("C", sae_c)]:
        model_path = os.path.join(args.output_dir, f"model_{name}_state_dict.pth")
        torch.save(model.state_dict(), model_path)
    print(f"Model weights saved to {args.output_dir}")

    print("\nDone! Run plot_similarity.py to generate visualizations.")


if __name__ == "__main__":
    main()
