import os
import json
import argparse
import torch
import glob
import signal
from .train import train_sae

# Local imports
import data
import sae_factory
import metric

# --- Configuration ---
CONFIG = {
    'num_saes': 4,
    'd_model': 2000,
    'k_fraction': 0.1,
    'epochs': 50,
    'batch_size': 32768,
    'prefetch_factor': 8,
    'num_workers': 8,
    'lr': 1e-4,
    'n_candidates': 32000,
    'delta': 1.0,
    'per_init': 0.05,
    'checkpoint_every_n_epochs': 2,
    'compile_model': False,
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


def get_checkpoint_dir(base_dir, model_type):
    """Get checkpoint directory for a given model type."""
    checkpoint_dir = os.path.join(base_dir, f"checkpoints_{model_type}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def is_training_complete(checkpoint_dir, sae_index, total_epochs):
    """Check if training for a specific SAE is already complete (NOT USED in eval-only mode)."""
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_sae_{sae_index}.pt')
    if not os.path.exists(checkpoint_path):
        return False
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    return checkpoint['epoch'] >= total_epochs - 1

def run_evaluation(model_type, args, device, loader, d_brain):
    """Core function to load a model and run its evaluation metrics."""
    print(f"\n{'='*60}")
    print(f"--- Starting Evaluation for Model Type: {model_type} ---")
    print(f"{'='*60}")

    k = int(CONFIG['k_fraction'] * CONFIG['d_model'])
    all_dictionaries = []
    all_results = {}
    
    # Determine which SAEs to evaluate
    if args.single_sae:
        sae_indices = [args.single_sae]
    else:
        sae_indices = list(range(1, CONFIG['num_saes'] + 1))
    
    for i in sae_indices:
        save_path = os.path.join(args.output_dir, f"sae_{i}__{model_type}_state_dict.pth")
        
        if not os.path.exists(save_path):
            print(f"[Warning] Trained model not found: {save_path}. Skipping SAE {i}.")
            continue
            
        print(f"\n--- Loading and Evaluating SAE {i}/{CONFIG['num_saes']} ---")
        
        # Initialize SAE model structure
        sae = sae_factory.get_sae_model(
            model_type, d_brain, CONFIG['d_model'], k, device, loader, CONFIG
        )
        
        # Load trained weights
        sae.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        sae.to(device)
        sae.eval() # Set to evaluation mode

        print("Running evaluation...")
        with torch.inference_mode():
            metrics, dictionary = metric.evaluate_sae(sae, loader, device)
        
        all_results[f"sae_{i}"] = metrics
        all_dictionaries.append(dictionary)
        print(f"SAE {i} Metrics: {json.dumps(metrics, indent=2)}")

    if not all_results:
        print(f"No completed SAEs found for {model_type}. Cannot aggregate results.")
        return

    print("\n--- Aggregating Results ---")
    avg_metrics = {}
    if all_results:
        metric_keys = list(all_results.values())[0].keys()
        for key in metric_keys:
            vals = [all_results[sae_key][key] for sae_key in all_results if key in all_results[sae_key]]
            if vals:
                avg_metrics[f"avg_{key}"] = sum(vals) / len(vals)

    stability_metrics = metric.compute_pairwise_stability(all_dictionaries, device)
    
    final_output = {
        "CONFIG": CONFIG,
        "model_type": model_type,
        "individual_averages": avg_metrics,
        "stability": stability_metrics,
        "raw": all_results
    }

    print("\n" + "="*60)
    print(f"FINAL RESULTS: {model_type}")
    print("="*60)
    print(json.dumps(final_output, indent=2))
    
    results_path = os.path.join(args.output_dir, f"final_results_{model_type}_eval_only.json")
    with open(results_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    return final_output


def eval_main(args):
    """New main function to run evaluation for all three models."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Evaluation on {device}")
    print(f"PyTorch version: {torch.__version__}")
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    shard_files = sorted(glob.glob(os.path.join(args.shard_directory, 'shard_*.pt')))
    if not shard_files:
        raise FileNotFoundError(f"No .pt shards found in {args.shard_directory}")
    
    print(f"Found {len(shard_files)} shard files")
        
    first_shard = torch.load(shard_files[0], map_location='cpu', weights_only=True)
    d_brain = first_shard.shape[-1]
    print(f"Detected embedding dimension: {d_brain}")

    mean, std = data.get_dataset_stats(args.shard_directory)
    normalizer = data.GPUNormalizer(mean, std).to(device)
    raw_loader = data.create_dataloader(
        args.shard_directory, 
        CONFIG['batch_size'], 
        num_workers=CONFIG['num_workers'], 
        prefetch_factor=CONFIG['prefetch_factor']
    )
    loader = data.DeviceDataLoader(raw_loader, device, normalizer)

    all_model_types = ["SAE", "RA-SAE", "SI-SAE"]
    master_results = {}
    
    for model_type in all_model_types:
        results = run_evaluation(model_type, args, device, loader, d_brain)
        if results:
            master_results[model_type] = results

    print("\n" + "#"*60)
    print("ALL EVALUATIONS COMPLETE")
    print("#"*60)

    # Save aggregated master results
    master_path = os.path.join(args.output_dir, f"master_eval_only_results.json")
    with open(master_path, "w") as f:
        json.dump(master_results, f, indent=2)
    print(f"\nSaved master aggregated results to {master_path}")


def main():
    parser = argparse.ArgumentParser(description="Train/Evaluate SAE with checkpointing support")
    parser.add_argument("shard_directory", type=str, help="Directory containing shard_*.pt files")
    parser.add_argument("model_type", type=str, choices=["SAE", "RA-SAE", "SI-SAE", "ALL"], nargs='?', default="ALL",
                        help="Model type to train. Use 'ALL' for eval-only mode.") # Changed
    parser.add_argument("--eval-only", action="store_true", 
                        help="Run evaluation for all three models (SAE, RA-SAE, SI-SAE) and exit.") # New flag
    parser.add_argument("--checkpoint-dir", type=str, default="../checkpoints",
                        help="Base directory for checkpoints")
    parser.add_argument("--output-dir", type=str, default="../trained_models",
                        help="Directory for final trained models")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--checkpoint-every", type=int, default=None,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for acceleration (PyTorch 2.0+)")
    parser.add_argument("--start-sae", type=int, default=1,
                        help="Start from SAE index (1-based)")
    parser.add_argument("--single-sae", type=int, default=None,
                        help="Train/Evaluate only this SAE index (1-based)")
    args = parser.parse_args()

    if args.eval_only:
        # Override model_type to "ALL" if eval-only is set
        args.model_type = "ALL" 
        eval_main(args)
        return

    # --- Original Training Logic Follows (Modified to use args.model_type for single training) ---
    if args.model_type == "ALL":
         # If no eval-only and model_type is ALL, raise error or default to one
         raise ValueError("Model type 'ALL' is only valid with the '--eval-only' flag.")
    
    # Rest of the original main() logic for single-model training
    if args.epochs:
        CONFIG['epochs'] = args.epochs
    if args.checkpoint_every:
        CONFIG['checkpoint_every_n_epochs'] = args.checkpoint_every
    if args.compile:
        CONFIG['compile_model'] = True

    killer = GracefulKiller()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running {args.model_type} on {device}")
    # ... (rest of original main logic to train a single model type)

    # --- Start of Original main() logic for single-model training ---
    # ... (omitted for brevity, as the user only requested the eval-only version)
    
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = get_checkpoint_dir(args.checkpoint_dir, args.model_type)
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Output directory: {args.output_dir}")

    shard_files = sorted(glob.glob(os.path.join(args.shard_directory, 'shard_*.pt')))
    if not shard_files:
        raise FileNotFoundError(f"No .pt shards found in {args.shard_directory}")
    
    print(f"Found {len(shard_files)} shard files")
        
    first_shard = torch.load(shard_files[0], map_location='cpu', weights_only=True)
    d_brain = first_shard.shape[-1]
    print(f"Detected embedding dimension: {d_brain}")

    mean, std = data.get_dataset_stats(args.shard_directory)
    normalizer = data.GPUNormalizer(mean, std).to(device)
    raw_loader = data.create_dataloader(
        args.shard_directory, 
        CONFIG['batch_size'], 
        num_workers=CONFIG['num_workers'], 
        prefetch_factor=CONFIG['prefetch_factor']
    )
    loader = data.DeviceDataLoader(raw_loader, device, normalizer)
    
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
        
        save_path = os.path.join(args.output_dir, f"sae_{i}__{args.model_type}_state_dict.pth")
        
        if os.path.exists(save_path) and is_training_complete(checkpoint_dir, i, CONFIG['epochs']):
            print(f"Loading existing completed model from {save_path}")
            sae = sae_factory.get_sae_model(
                args.model_type, d_brain, CONFIG['d_model'], k, device, loader, CONFIG
            )
            sae.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        else:
            print("Training (will resume from checkpoint if available)...")
            
            sae = sae_factory.get_sae_model(
                args.model_type, d_brain, CONFIG['d_model'], k, device, loader, CONFIG
            )
            
            optimizer = torch.optim.Adam(sae.parameters(), lr=CONFIG['lr'])
            num_step_epoch = 250_000_000 // CONFIG['batch_size']
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.001, end_factor=1, total_iters=num_step_epoch * 3
            )

            def criterion(x, x_hat, *args): 
                return (x - x_hat).square().mean()
            
            logs = train_sae(
                sae, loader, criterion, optimizer,
                scheduler=scheduler,
                nb_epochs=CONFIG['epochs'],
                device=device,
                sae_index=i,
                monitoring=2,
                checkpoint_dir=checkpoint_dir,
                checkpoint_every_n_epochs=CONFIG['checkpoint_every_n_epochs'],
                compile_model=CONFIG['compile_model']
            )
            
            torch.save(sae.state_dict(), save_path)
            print(f"Saved final model to {save_path}")

        
        print("Running evaluation...")
        with torch.inference_mode():
            metrics, dictionary = metric.evaluate_sae(sae, loader, device)
        
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
        partial_path = os.path.join(args.output_dir, f"partial_results_{args.model_type}.json")
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

    stability_metrics = metric.compute_pairwise_stability(all_dictionaries, device)
    
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
    
    results_path = os.path.join(args.output_dir, f"final_results_{args.model_type}.json")
    with open(results_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved results to {results_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()