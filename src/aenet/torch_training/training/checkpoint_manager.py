"""
Checkpoint management for PyTorch training.

Handles saving, loading, and rotating model checkpoints during training.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class CheckpointManager:
    """
    Manages checkpoint operations for training.

    Handles:
    - Saving checkpoints with full training state
    - Loading checkpoints to resume training
    - Rotating old checkpoints to limit disk usage
    - Tracking best model based on validation metrics

    Parameters
    ----------
    checkpoint_dir : str or Path, optional
        Directory to save checkpoints. If None, checkpoints are disabled.
    max_to_keep : int, optional
        Maximum number of checkpoints to keep. Older ones are deleted.
    save_best : bool
        Whether to save the best model separately.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        max_to_keep: Optional[int] = None,
        save_best: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.max_to_keep = max_to_keep
        self.save_best = save_best
        self.best_val_loss: Optional[float] = None

        if self.checkpoint_dir is not None:
            self._ensure_dir(self.checkpoint_dir)

    def _ensure_dir(self, path: Path):
        """Create directory if it doesn't exist."""
        path.mkdir(parents=True, exist_ok=True)

    def _rotate_checkpoints(self):
        """Remove old checkpoints beyond max_to_keep limit."""
        if self.checkpoint_dir is None or self.max_to_keep is None:
            return
        if self.max_to_keep <= 0:
            return

        ckpts = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(ckpts) <= self.max_to_keep:
            return

        to_remove = ckpts[: len(ckpts) - self.max_to_keep]
        for p in to_remove:
            try:
                p.unlink()
            except Exception:
                pass

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        history: Dict[str, Any],
        architecture: Dict[str, Any],
        descriptor_config: Dict[str, Any],
        filename: Optional[str] = None,
    ):
        """
        Save a training checkpoint.

        Parameters
        ----------
        model : nn.Module
            Model to save.
        optimizer : torch.optim.Optimizer
            Optimizer to save.
        epoch : int
            Current epoch number.
        history : dict
            Training history.
        architecture : dict
            Network architecture specification.
        descriptor_config : dict
            Descriptor configuration.
        filename : str, optional
            Filename for checkpoint. If None, uses format
            "checkpoint_epoch_{epoch:04d}.pt"
        """
        if self.checkpoint_dir is None:
            return

        if filename is None:
            filename = f"checkpoint_epoch_{epoch:04d}.pt"

        path = self.checkpoint_dir / filename

        payload = {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "architecture": architecture,
            "descriptor_config": descriptor_config,
            "best_val_loss": (
                float(self.best_val_loss)
                if self.best_val_loss is not None
                else None
            ),
        }

        try:
            torch.save(payload, str(path))
        except Exception as e:
            print(f"[WARN] Failed to save checkpoint at {path}: {e}")

    def load_checkpoint(
        self,
        path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint and restore training state.

        Parameters
        ----------
        path : str
            Path to checkpoint file.
        model : nn.Module
            Model to load state into.
        optimizer : torch.optim.Optimizer
            Optimizer to load state into.
        device : torch.device
            Device to map tensors to.

        Returns
        -------
        dict
            Checkpoint payload containing epoch, history, etc.

        Raises
        ------
        RuntimeError
            If checkpoint loading fails.
        """
        try:
            payload = torch.load(path, map_location=device)
            model.load_state_dict(payload["model_state_dict"])
            optimizer.load_state_dict(payload["optimizer_state_dict"])

            if "best_val_loss" in payload:
                self.best_val_loss = payload["best_val_loss"]

            return payload
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint '{path}': {e}")

    def should_save_best(self, val_loss: float) -> bool:
        """
        Check if current validation loss is the best so far.

        Parameters
        ----------
        val_loss : float
            Current validation loss.

        Returns
        -------
        bool
            True if this is the best validation loss.
        """
        if not self.save_best:
            return False

        is_best = (
            (self.best_val_loss is None) or (val_loss < self.best_val_loss)
        )
        if is_best:
            self.best_val_loss = float(val_loss)
        return is_best

    def save_best_model(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        history: Dict[str, Any],
        architecture: Dict[str, Any],
        descriptor_config: Dict[str, Any],
    ):
        """Save the best model checkpoint."""
        if self.checkpoint_dir is None or not self.save_best:
            return

        self.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            history=history,
            architecture=architecture,
            descriptor_config=descriptor_config,
            filename="best_model.pt",
        )

    def infer_start_epoch(self, checkpoint_path: str) -> int:
        """
        Infer starting epoch from checkpoint filename.

        Parameters
        ----------
        checkpoint_path : str
            Path to checkpoint file.

        Returns
        -------
        int
            Starting epoch (checkpoint epoch + 1), or 0 if cannot infer.
        """
        try:
            name = Path(checkpoint_path).name
            if name.startswith("checkpoint_epoch_") and name.endswith(".pt"):
                return int(name[len("checkpoint_epoch_"): -3]) + 1
        except Exception:
            pass
        return 0
