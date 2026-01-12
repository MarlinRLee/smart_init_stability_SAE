import time
import os
from collections import defaultdict

import torch
from einops import rearrange

from overcomplete.metrics import l2, r2_score, l0_eps
from overcomplete.sae.trackers import DeadCodeTracker
from overcomplete.sae.train import extract_input


def _compute_reconstruction_error(x, x_hat):
    """
    Try to match the shapes of x and x_hat to compute the reconstruction error.
    
    Ensures both tensors are float32 for consistent computation.
    """
    # Ensure consistent dtype for accurate comparison
    x = x.float()
    x_hat = x_hat.float()
    
    if len(x.shape) == 4 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, 'n c w h -> (n w h) c')
    elif len(x.shape) == 3 and len(x_hat.shape) == 2:
        x_flatten = rearrange(x, 'n t c -> (n t) c')
    else:
        assert x.shape == x_hat.shape, "Input and output shapes must match."
        x_flatten = x

    r2 = r2_score(x_flatten, x_hat)
    return r2.item()


def _log_metrics(monitoring, logs, model, z, loss, optimizer, current_step, sae_prefix):
    """Log training metrics for the current training step."""
    if monitoring == 0:
        return

    if monitoring > 0:
        lr = optimizer.param_groups[0]['lr']
        step_loss = loss.item()
        logs['lr'].append(lr)
        logs['step_loss'].append(step_loss)

    if monitoring > 1:
        z_l2 = l2(z.float()).item()
        dictionary_sparsity = l0_eps(model.get_dictionary()).mean().item()
        dictionary_norms = l2(model.get_dictionary(), -1).mean().item()
        
        logs['z_l2'].append(z_l2)
        logs['dictionary_sparsity'].append(dictionary_sparsity)
        logs['dictionary_norms'].append(dictionary_norms)


def save_checkpoint(checkpoint_dir, sae_index, epoch, global_step, model, optimizer, 
                    scheduler, logs, best_loss=None, early_stopping_state=None):
    """Save a training checkpoint for resuming later."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'logs': dict(logs),
        'best_loss': best_loss,
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if early_stopping_state is not None:
        checkpoint['early_stopping_state'] = early_stopping_state
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_sae_{sae_index}.pt')
    temp_path = checkpoint_path + '.tmp'
    
    # Save to temp file first, then rename (atomic operation)
    torch.save(checkpoint, temp_path)
    os.replace(temp_path, checkpoint_path)
    
    print(f"  [Checkpoint] Saved at epoch {epoch+1}, step {global_step}")
    return checkpoint_path


def load_checkpoint(checkpoint_dir, sae_index, model, optimizer, scheduler=None, device='cuda'):
    """Load a training checkpoint if it exists."""
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_sae_{sae_index}.pt')
    
    if not os.path.exists(checkpoint_path):
        return None
    
    print(f"  [Checkpoint] Loading from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    logs = defaultdict(list, checkpoint.get('logs', {}))
    
    print(f"  [Checkpoint] Resuming from epoch {checkpoint['epoch']+1}, step {checkpoint['global_step']}")
    
    return {
        'epoch': checkpoint['epoch'],
        'global_step': checkpoint['global_step'],
        'logs': logs,
        'best_loss': checkpoint.get('best_loss'),
        'early_stopping_state': checkpoint.get('early_stopping_state'),
    }


class EarlyStopping:
    """
    Early stopping handler to stop training when validation loss stops improving.
    
    Parameters
    ----------
    patience : int
        Number of epochs to wait for improvement before stopping.
    min_delta : float
        Minimum change in validation loss to qualify as an improvement.
    mode : str
        'min' for loss (lower is better), 'max' for metrics like R2 (higher is better).
    """
    def __init__(self, patience=5, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        """
        Check if training should stop.
        
        Returns
        -------
        bool
            True if this is a new best score, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return True
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False
    
    def get_state(self):
        """Get state for checkpointing."""
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop,
            'best_epoch': self.best_epoch,
        }
    
    def load_state(self, state):
        """Load state from checkpoint."""
        if state is not None:
            self.counter = state['counter']
            self.best_score = state['best_score']
            self.early_stop = state['early_stop']
            self.best_epoch = state['best_epoch']


def validate(model, val_loader, criterion, device):
    """Run validation on the validation set."""
    model.eval()
    
    total_loss = 0.0
    total_r2 = 0.0
    total_l0 = 0.0
    total_samples = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            x = extract_input(batch).to(device, non_blocking=True)
            x = x.float()
            
            z_pre, z, x_hat = model(x)
            loss = criterion(x, x_hat, z_pre, z, model.get_dictionary())
            
            total_loss += loss.item()
            total_r2 += _compute_reconstruction_error(x, x_hat)
            total_l0 += l0_eps(z, 0).sum().item()
            total_samples += x.shape[0]
            batch_count += 1
    
    model.train()
    
    return {
        'val_loss': total_loss / batch_count if batch_count > 0 else float('inf'),
        'val_r2': total_r2 / batch_count if batch_count > 0 else 0.0,
        'val_l0': total_l0 / batch_count if batch_count > 0 else 0.0,  # Per-sample average
    }


def train_sae(model, dataloader, criterion, optimizer, scheduler=None,
              nb_epochs=20, clip_grad=1.0, monitoring=1, device="cpu", sae_index=1,
              checkpoint_dir=None, checkpoint_every_n_epochs=1, model_type=None,
              val_loader=None, early_stopping_patience=None, early_stopping_min_delta=0.0,
              use_mixed_precision=False):
    """
    Train a Sparse Autoencoder (SAE) model with checkpointing support.
    
    Parameters
    ----------
    model : nn.Module
        The SAE model to train.
    dataloader : DataLoader
        Training data loader.
    criterion : callable
        Loss function.
    optimizer : Optimizer
        Optimizer for training.
    scheduler : LRScheduler, optional
        Learning rate scheduler.
    nb_epochs : int
        Maximum number of epochs to train.
    clip_grad : float
        Gradient clipping value.
    monitoring : int
        Level of monitoring (0=none, 1=basic, 2=detailed).
    device : str
        Device to train on.
    sae_index : int
        Index of the SAE (for checkpointing).
    checkpoint_dir : str, optional
        Directory to save checkpoints.
    checkpoint_every_n_epochs : int
        Save checkpoint every N epochs.
    model_type : str, optional
        Type of model (used for SI-SAE freezing).
    val_loader : DataLoader, optional
        Validation data loader. If provided, validation will be run each epoch.
    early_stopping_patience : int, optional
        If provided, enables early stopping with this patience value.
    early_stopping_min_delta : float
        Minimum improvement required for early stopping.
    use_mixed_precision : bool
        If True, use bfloat16 mixed precision training (recommended for A100).
    
    Returns
    -------
    dict
        Training logs.
    """
    logs = defaultdict(list)
    global_step = 0
    start_epoch = 0
    best_loss = float('inf')
    best_val_loss = float('inf')
    sae_prefix = f"sae_{sae_index}"
    
    # Mixed precision setup
    if use_mixed_precision:
        if not torch.cuda.is_available():
            print("  [Warning] Mixed precision requested but CUDA not available. Disabling.")
            use_mixed_precision = False
        elif not torch.cuda.is_bf16_supported():
            print("  [Warning] BF16 not supported on this GPU. Disabling mixed precision.")
            use_mixed_precision = False
        else:
            print("  [Mixed Precision] Using bfloat16 (no loss scaling needed)")
    
    # Initialize early stopping if requested
    early_stopper = None
    if early_stopping_patience is not None and val_loader is not None:
        early_stopper = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            mode='min'  # We're tracking validation loss
        )
    
    # Try to resume from checkpoint
    if checkpoint_dir:
        resumed = load_checkpoint(checkpoint_dir, sae_index, model, optimizer, scheduler, device=device)
        if resumed:
            start_epoch = resumed['epoch'] + 1
            global_step = resumed['global_step']
            logs = resumed['logs']
            best_loss = resumed.get('best_loss', float('inf'))
            
            # Restore early stopping state
            if early_stopper is not None and resumed.get('early_stopping_state'):
                early_stopper.load_state(resumed['early_stopping_state'])
                best_val_loss = early_stopper.best_score if early_stopper.best_score else float('inf')
            
            if start_epoch >= nb_epochs:
                print(f"  [Checkpoint] Training already complete for {sae_prefix}")
                return logs

    print(f"Starting training for {sae_prefix} from epoch {start_epoch + 1}")
    print(f"  Mixed precision: {'bfloat16' if use_mixed_precision else 'disabled (fp32)'}")
    
    if val_loader is not None:
        print(f"  Validation enabled")
        if early_stopper is not None:
            print(f"  Early stopping enabled (patience={early_stopping_patience}, min_delta={early_stopping_min_delta})")

    frozen = False
    if model_type == "SI-SAE" and start_epoch < 10:
        for param in model.dictionary.parameters():
            param.requires_grad = False
        frozen = True
            
            
    for epoch in range(start_epoch, nb_epochs):
        if frozen and epoch >= 10:
            for param in model.dictionary.parameters():
                param.requires_grad = True
            print("unfreeze dict", flush = True)
            frozen = False
            
        model.train()

        start_time = time.time()
        epoch_loss = 0.0
        epoch_error = 0.0
        epoch_sparsity = 0.0
        batch_count = 0
        mon_count = 0
        dead_tracker = None

        for batch in dataloader:
            global_step += 1
            batch_count += 1
            
            x = extract_input(batch).to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)

            if use_mixed_precision:
                # === MIXED PRECISION TRAINING ===
                # Forward pass in bfloat16
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    z_pre, z, x_hat = model(x)
                
                # Compute loss in fp32 for numerical stability
                # This is the key to avoiding NaN/Inf issues
                loss = criterion(
                    x.float(), 
                    x_hat.float(), 
                    z_pre.float(), 
                    z.float(), 
                    model.get_dictionary().float()
                )
                
                # Dead tracker needs float
                if dead_tracker is None:
                    dead_tracker = DeadCodeTracker(z.shape[1], device)
                dead_tracker.update(z.float())
                
                # Backward pass - gradients computed in mixed precision
                # but accumulated in fp32 (PyTorch handles this automatically)
                loss.backward()
                
            else:
                # === STANDARD FP32 TRAINING ===
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

            if monitoring and batch_count % 50 == 0:
                mon_count += 1
                epoch_loss += loss.item()
                epoch_error += _compute_reconstruction_error(x, x_hat)
                epoch_sparsity += l0_eps(z.float(), 0).sum().item()
                _log_metrics(monitoring, logs, model, z.float(), loss, optimizer, global_step, sae_prefix) 

        epoch_duration = time.time() - start_time
        
        # Training metrics
        if monitoring and batch_count > 0 and mon_count > 0:
            avg_loss = epoch_loss / mon_count
            avg_error = epoch_error / mon_count
            avg_sparsity = epoch_sparsity / mon_count
            dead_ratio = dead_tracker.get_dead_ratio()

            logs['avg_loss'].append(avg_loss)
            logs['r2'].append(avg_error)
            logs['time_epoch'].append(epoch_duration)
            logs['z_sparsity'].append(avg_sparsity)
            logs['dead_features'].append(dead_ratio)
            
            if avg_loss < best_loss:
                best_loss = avg_loss

            train_msg = (f"Epoch[{epoch+1}/{nb_epochs}] Train - Loss: {avg_loss:.4f}, "
                        f"R2: {avg_error:.4f}, L0: {avg_sparsity:.4f}, "
                        f"Dead: {dead_ratio*100:.1f}%, Time: {epoch_duration:.2f}s")
        else:
            train_msg = f"Epoch[{epoch+1}/{nb_epochs}] Time: {epoch_duration:.2f}s"
        
        # Validation
        val_msg = ""
        if val_loader is not None:
            val_metrics = validate(model, val_loader, criterion, device)
            
            logs['val_loss'].append(val_metrics['val_loss'])
            logs['val_r2'].append(val_metrics['val_r2'])
            logs['val_l0'].append(val_metrics['val_l0'])
            
            val_msg = (f" | Val - Loss: {val_metrics['val_loss']:.4f}, "
                      f"R2: {val_metrics['val_r2']:.4f}, L0: {val_metrics['val_l0']:.4f}")
            
            # Early stopping check
            if early_stopper is not None:
                is_best = early_stopper(val_metrics['val_loss'], epoch)
                if is_best:
                    best_val_loss = val_metrics['val_loss']
                    val_msg += " *best*"
                    # Save best model
                    if checkpoint_dir:
                        best_path = os.path.join(checkpoint_dir, f'best_sae_{sae_index}.pt')
                        torch.save(model.state_dict(), best_path)
                else:
                    val_msg += f" (no improv. {early_stopper.counter}/{early_stopper.patience})"
        
        print(train_msg + val_msg)
        
        # Check for early stopping
        if early_stopper is not None and early_stopper.early_stop:
            print(f"\n  [Early Stopping] No improvement for {early_stopper.patience} epochs. "
                  f"Best val_loss: {early_stopper.best_score:.4f} at epoch {early_stopper.best_epoch + 1}")
            
            # Load best model if available
            if checkpoint_dir:
                best_path = os.path.join(checkpoint_dir, f'best_sae_{sae_index}.pt')
                if os.path.exists(best_path):
                    print(f"  [Early Stopping] Loading best model from epoch {early_stopper.best_epoch + 1}")
                    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
            break

        # Save checkpoint
        if checkpoint_dir and (epoch + 1) % checkpoint_every_n_epochs == 0:
            early_stopping_state = early_stopper.get_state() if early_stopper else None
            save_checkpoint(checkpoint_dir, sae_index, epoch, global_step, 
                          model, optimizer, scheduler, logs, best_loss, early_stopping_state)

    # Final checkpoint (only if not early stopped)
    if checkpoint_dir and (early_stopper is None or not early_stopper.early_stop):
        early_stopping_state = early_stopper.get_state() if early_stopper else None
        save_checkpoint(checkpoint_dir, sae_index, nb_epochs - 1, global_step,
                       model, optimizer, scheduler, logs, best_loss, early_stopping_state)

    # Summary
    if val_loader is not None:
        print(f"\n  Training Summary for {sae_prefix}:")
        print(f"    Final train loss: {logs['avg_loss'][-1]:.4f}" if logs['avg_loss'] else "")
        print(f"    Best val loss: {best_val_loss:.4f}")
        if early_stopper:
            print(f"    Best epoch: {early_stopper.best_epoch + 1}")
            print(f"    Stopped at epoch: {epoch + 1}")

    return logs