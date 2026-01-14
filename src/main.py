import os
import json
import argparse
import torch
import glob
import signal

# Local imports
from .data import get_dataset_stats, GPUNormalizer, create_dataloader, DeviceDataLoader, create_val_dataloader
from .sae_factory import get_sae_model
from .metric import evaluate_sae, compute_pairwise_stability
from .train import train_sae


# --- Configuration ---
CONFIG = {
    'num_saes': 2,
    'd_model': 10_000,
    'k_fraction': 0.01,
    'epochs': 20,
    'batch_size': 16_384*2,
    'prefetch_factor': 2,
    'num_workers': 8,
    'lr': 1e-3,
    #RA SAE:
    'n_candidates': 32_000,
    'delta': 0.1,
    #SI SAE
    'per_init': 0.0,
    
    #Meta or validation
    'checkpoint_every_n_epochs': 5,
    # Validation and early stopping settings
    'val_batch_size': 16_384,  # Batch size for validation
    'val_num_workers': 4,      # Workers for validation loader
    'val_subset_fraction': 1.0, # Use all validation data (or reduce for speed)
    'early_stopping_patience': 10,  # Stop if no improvement for N epochs
    'early_stopping_min_delta': 0,  # Minimum improvement to count
}


class GracefulKiller:
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    kill_now = False
    
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    
    def exit_gracefully(self, signum, frame):
        print("\n[Signal] Received shutdown signal. Will save checkpoint after current epoch...")
        self.kill_now = True

def get_checkpoint_dir(base_dir, model_type, run_suffix):
    """Get checkpoint directory with specific model parameters."""
    checkpoint_dir = os.path.join(base_dir, f"checkpoints_{model_type}{run_suffix}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return checkpoint_dir


def is_training_complete(checkpoint_dir, sae_index, total_epochs):
    """Check if training for a specific SAE is already complete."""
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_sae_{sae_index}.pt')
    if not os.path.exists(checkpoint_path):
        return False
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return checkpoint['epoch'] >= total_epochs - 1

def main():
    parser = argparse.ArgumentParser(description="Train SAE with checkpointing support")
    parser.add_argument("shard_directory", type=str, help="Directory containing shard_*.pt files")
    parser.add_argument("model_type", type=str, choices=["SAE", "RA-SAE", "SI-SAE"])
    parser.add_argument("--checkpoint-dir", type=str, default="../checkpoints",
                        help="Base directory for checkpoints")
    parser.add_argument("--output-dir", type=str, default="../trained_models",
                        help="Directory for final trained models")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--start-sae", type=int, default=1,
                        help="Start from SAE index (1-based)")
    parser.add_argument("--single-sae", type=int, default=None,
                        help="Train only this SAE index (1-based)")
    
    # Validation arguments
    parser.add_argument("--val-dir", type=str, default=None,
                        help="Directory containing validation shard_*.pt files. "
                             "If not provided, validation is disabled.")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Disable early stopping even if validation is enabled")
    parser.add_argument("--early-stopping-patience", type=int, default=None,
                        help="Override early stopping patience")
    parser.add_argument("--val-subset", type=float, default=None,
                        help="Fraction of validation shards to use (0.0-1.0)")
    
    # Performance arguments
    parser.add_argument("--mixed-precision", action="store_true",
                        help="Use bfloat16 mixed precision training (recommended for A100)")
    
    args = parser.parse_args()

    if args.epochs:
        CONFIG['epochs'] = args.epochs
    if args.checkpoint_every:
        CONFIG['checkpoint_every_n_epochs'] = args.checkpoint_every
    if args.early_stopping_patience:
        CONFIG['early_stopping_patience'] = args.early_stopping_patience
    if args.val_subset:
        CONFIG['val_subset_fraction'] = args.val_subset

    killer = GracefulKiller()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Enable TF32 for free speedup on Ampere+ GPUs (A100, etc.)
    # TF32 uses tensor cores with minimal precision impact
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled for matrix multiplications")
    
    print(f"Running {args.model_type} on {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.output_dir, exist_ok=True)
    
    k = int(CONFIG['k_fraction'] * CONFIG['d_model'])
    run_suffix = f"_d{CONFIG['d_model']}_k{k}"
    if args.model_type == "RA-SAE":
        run_suffix += f"_delta{CONFIG['delta']}"
    elif args.model_type == "SI-SAE":
        run_suffix += f"_per_init{CONFIG['per_init']}"

    # Update checkpoint directory call to use the suffix
    checkpoint_dir = get_checkpoint_dir(args.checkpoint_dir, args.model_type, run_suffix)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Output directory: {args.output_dir}")

    # --- Training Data ---
    shard_files = sorted(glob.glob(os.path.join(args.shard_directory, 'shard_*.pt')))
    if not shard_files:
        raise FileNotFoundError(f"No .pt shards found in {args.shard_directory}")
    
    print(f"Found {len(shard_files)} training shard files")
        
    first_shard = torch.load(shard_files[0], map_location='cpu', weights_only=True)
    d_brain = first_shard.shape[-1]
    print(f"Detected embedding dimension: {d_brain}")

    mean, std = get_dataset_stats(args.shard_directory)
    normalizer = GPUNormalizer(mean, std).to(device)
    raw_loader = create_dataloader(
        args.shard_directory, 
        CONFIG['batch_size'], 
        num_workers=CONFIG['num_workers'], 
        prefetch_factor=CONFIG['prefetch_factor']
    )
    loader = DeviceDataLoader(raw_loader, device, normalizer)
    
    # --- Validation Data ---
    val_loader = None
    early_stopping_patience = None
    
    if args.val_dir:
        val_shard_files = sorted(glob.glob(os.path.join(args.val_dir, 'shard_*.pt')))
        if not val_shard_files:
            print(f"[Warning] No validation shards found in {args.val_dir}. Disabling validation.")
        else:
            print(f"Found {len(val_shard_files)} validation shard files")
            
            # Try to get validation stats, fall back to training stats
            try:
                val_mean, val_std = get_dataset_stats(args.val_dir)
                val_normalizer = GPUNormalizer(val_mean, val_std).to(device)
                print("Using validation-specific normalization stats")
            except FileNotFoundError:
                print("No validation stats found, using training stats for normalization")
                val_normalizer = normalizer
            
            raw_val_loader = create_val_dataloader(
                args.val_dir,
                CONFIG['val_batch_size'],
                num_workers=CONFIG['val_num_workers'],
                prefetch_factor=2,
                subset_fraction=CONFIG['val_subset_fraction']
            )
            val_loader = DeviceDataLoader(raw_val_loader, device, val_normalizer)
            print(f"Validation enabled with {len(val_shard_files)} shards "
                  f"(using {CONFIG['val_subset_fraction']*100:.0f}%)")
            
            # Enable early stopping unless explicitly disabled
            if not args.no_early_stopping:
                early_stopping_patience = CONFIG['early_stopping_patience']
                print(f"Early stopping enabled (patience={early_stopping_patience})")
            else:
                print("Early stopping disabled by user")
    else:
        print("No validation directory provided. Training without validation.")

    k = int(CONFIG['k_fraction'] * CONFIG['d_model'])
    print(f"Config: d_model={CONFIG['d_model']}, k={k}, epochs={CONFIG['epochs']}")
    
    all_dictionaries = []
    all_results = {}
    
    if args.single_sae:
        sae_indices = [args.single_sae]
    else:
        sae_indices = list(range(args.start_sae, CONFIG['num_saes'] + 1))
    
    for i in sae_indices:
        if killer.kill_now:
            print("[Signal] Stopping before next SAE due to shutdown signal")
            break
            
        print(f"\n{'='*60}")
        print(f"--- SAE {i}/{CONFIG['num_saes']} ---")
        print(f"{'='*60}")
        
        model_id = f"sae_{i}_{args.model_type}{run_suffix}"
        save_path = os.path.join(args.output_dir, f"{model_id}_state_dict.pth")
        
        if os.path.exists(save_path) and is_training_complete(checkpoint_dir, i, CONFIG['epochs']):
            print(f"Loading existing completed model from {save_path}")
            sae = get_sae_model(
                args.model_type, d_brain, CONFIG['d_model'], k, device, loader, CONFIG
            )
            sae.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        else:
            print("Training (will resume from checkpoint if available)...")
            
            sae = get_sae_model(
                args.model_type, d_brain, CONFIG['d_model'], k, device, loader, CONFIG
            )
            
            optimizer = torch.optim.Adam(sae.parameters(), lr=CONFIG['lr'])
            num_step_epoch = 250_000_000 // CONFIG['batch_size']
            
            total_steps = num_step_epoch * CONFIG['epochs']
            warmup_steps = num_step_epoch * 15
            
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=0.001, 
                end_factor=1.0, 
                total_iters=warmup_steps
            )

            # 3. Define the Decay Phase
            decay = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=(total_steps - warmup_steps), 
                eta_min=CONFIG['lr'] * 0.05
            )

            # 4. Chain them together
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, 
                schedulers=[warmup, decay], 
                milestones=[warmup_steps]
            )

            def criterion(x, x_hat, pre_codes, codes, dictionary):
                loss = (x - x_hat).square().mean()

                # add reanimation loss to avoid dead codes:
                is_dead = ((codes > 0).sum(dim=0) == 0).float().detach()
                reanim_loss = (pre_codes * is_dead[None, :]).mean()
                loss -= reanim_loss * 1e-3

                return loss
            
            logs = train_sae(
                sae, loader, criterion, optimizer,
                scheduler=scheduler,
                nb_epochs=CONFIG['epochs'],
                device=device,
                sae_index=i,
                monitoring=1,
                checkpoint_dir=checkpoint_dir,
                checkpoint_every_n_epochs=CONFIG['checkpoint_every_n_epochs'],
                model_type=args.model_type,
                # Validation parameters
                val_loader=val_loader,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=CONFIG['early_stopping_min_delta'],
                # Performance
                use_mixed_precision=args.mixed_precision
            )
            
            torch.save(sae.state_dict(), save_path)
            print(f"Saved final model to {save_path}")

        
        print("Running evaluation...")
        with torch.inference_mode():
            metrics, dictionary = evaluate_sae(sae, loader, device)
        
        all_results[f"sae_{i}"] = metrics
        all_dictionaries.append(dictionary)
        print(f"SAE {i} Metrics: {json.dumps(metrics, indent=2)}")

    if len(all_results) < CONFIG['num_saes']:
        print(f"\n[Warning] Only completed {len(all_results)}/{CONFIG['num_saes']} SAEs")
        if killer.kill_now:
            print("Training was interrupted. Run again to continue.")
        
        partial_output = {
            "CONFIG": CONFIG,
            "completed_saes": list(all_results.keys()),
            "raw": all_results
        }
        partial_path = os.path.join(args.output_dir, f"partial_results_{args.model_type}{run_suffix}.json")
        with open(partial_path, "w") as f:
            json.dump(partial_output, f, indent=2)
        print(f"Saved partial results to {partial_path}")
        return

    print("\n--- Aggregating Results ---")
    avg_metrics = {}
    metric_keys = all_results["sae_1"].keys()
    for key in metric_keys:
        vals = [all_results[f"sae_{j}"][key] for j in range(1, CONFIG['num_saes'] + 1)]
        avg_metrics[f"avg_{key}"] = sum(vals) / len(vals)

    stability_metrics = compute_pairwise_stability(all_dictionaries, device)
    
    final_output = {
        "CONFIG": CONFIG,
        "individual_averages": avg_metrics,
        "stability": stability_metrics,
        "raw": all_results
    }

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(json.dumps(final_output, indent=2))
    
    results_path = os.path.join(args.output_dir, f"final_results_{args.model_type}{run_suffix}.json")
    with open(results_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved results to {results_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()