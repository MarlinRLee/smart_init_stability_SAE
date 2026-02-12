import os
import signal
import torch


class GracefulKiller:
    """Handle SIGTERM/SIGINT for graceful shutdown."""

    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, signum, frame):
        print("\n[Signal] Received shutdown signal. Will save checkpoint after current epoch...")
        self.kill_now = True


def get_checkpoint_dir(base_dir, model_type, run_suffix=""):
    """Get checkpoint directory with optional model parameters suffix."""
    checkpoint_dir = os.path.join(base_dir, f"checkpoints_{model_type}{run_suffix}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def is_training_complete(checkpoint_dir, sae_index, total_epochs):
    """Check if training for a specific SAE is already complete.

    Returns True if either:
    - The checkpoint was explicitly marked as complete (normal finish or early stop)
    - The checkpoint epoch reached the final epoch (backward compat)
    """
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_sae_{sae_index}.pt')
    if not os.path.exists(checkpoint_path):
        return False

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('training_complete', False):
        return True
    return checkpoint['epoch'] >= total_epochs - 1
