"""
Optimizer and scheduler builder for PyTorch training.

Handles construction of optimizers and learning rate schedulers from
training configuration.
"""

from typing import Optional

import torch
import torch.nn as nn

from ..config import TrainingMethod, Adam, SGD


class OptimizerBuilder:
    """
    Builds optimizers and learning rate schedulers.

    Parameters
    ----------
    model : nn.Module
        Model whose parameters will be optimized.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def build_optimizer(self, method: TrainingMethod) -> torch.optim.Optimizer:
        """
        Build optimizer from training method configuration.

        Parameters
        ----------
        method : TrainingMethod
            Training method configuration (Adam or SGD).

        Returns
        -------
        torch.optim.Optimizer
            Configured optimizer.
        """
        params = self.model.parameters()

        if isinstance(method, Adam):
            return torch.optim.Adam(
                params,
                lr=float(method.mu),
                betas=(float(method.beta1), float(method.beta2)),
                eps=float(method.epsilon),
                weight_decay=float(method.weight_decay),
            )
        elif isinstance(method, SGD):
            return torch.optim.SGD(
                params,
                lr=float(method.lr),
                momentum=float(method.momentum),
                weight_decay=float(method.weight_decay),
            )
        else:
            # Default to Adam with conservative params if unknown
            return torch.optim.Adam(params, lr=1e-3)

    @staticmethod
    def build_scheduler(
        optimizer: torch.optim.Optimizer,
        use_scheduler: bool = False,
        scheduler_patience: int = 10,
        scheduler_factor: float = 0.5,
        scheduler_min_lr: float = 1e-6,
    ) -> Optional[torch.optim.lr_scheduler.ReduceLROnPlateau]:
        """
        Build learning rate scheduler.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer to schedule.
        use_scheduler : bool
            Whether to use a scheduler.
        scheduler_patience : int
            Number of epochs with no improvement after which LR is reduced.
        scheduler_factor : float
            Factor by which LR is reduced.
        scheduler_min_lr : float
            Minimum learning rate.

        Returns
        -------
        torch.optim.lr_scheduler.ReduceLROnPlateau or None
            Scheduler if use_scheduler is True, otherwise None.
        """
        if not use_scheduler:
            return None

        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            min_lr=scheduler_min_lr,
        )

    @staticmethod
    def get_batch_size(method: TrainingMethod) -> int:
        """
        Extract batch size from training method.

        Parameters
        ----------
        method : TrainingMethod
            Training method configuration.

        Returns
        -------
        int
            Batch size (defaults to 32 if not found).
        """
        if hasattr(method, "batchsize"):
            return int(getattr(method, "batchsize"))
        if hasattr(method, "batch_size"):
            return int(getattr(method, "batch_size"))
        return 32
