"""
Training loop execution for PyTorch training.

Handles epoch execution including batch processing, loss computation,
and optimization steps.
"""

import time
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from ..loss import compute_energy_loss, compute_force_loss
from .normalization import NormalizationManager

# Progress bar (match aenet.mlip behavior)
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore


def _iter_progress(iterable, enable: bool, desc: str):
    """Wrap an iterable with tqdm progress bar if enabled and available."""
    if enable and tqdm is not None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except Exception:
            total = None
        return tqdm(iterable, total=total, desc=desc, ncols=80, leave=False)
    return iterable


class TrainingLoop:
    """
    Executes training and validation epochs.

    Handles:
    - Batch iteration and processing
    - Loss computation (energy + forces)
    - Backpropagation and optimization
    - Timing and metrics collection

    Parameters
    ----------
    model : nn.Module
        Model to train (EnergyModelAdapter).
    descriptor : ChebyshevDescriptor
        Descriptor for force computation.
    normalizer : NormalizationManager
        Normalization manager for features and energies.
    device : torch.device
        Device for computation.
    dtype : torch.dtype
        Data type for tensors.
    """

    def __init__(
        self,
        model,
        descriptor,
        normalizer: NormalizationManager,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.model = model
        self.descriptor = descriptor
        self.normalizer = normalizer
        self.device = device
        self.dtype = dtype

        # Timing state (last epoch)
        self.last_forward_time: float = 0.0
        self.last_backward_time: float = 0.0
        self.last_data_loading_time: float = 0.0
        self.last_loss_computation_time: float = 0.0
        self.last_optimizer_time: float = 0.0

    def run_epoch(
        self,
        loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer],
        alpha: float,
        energy_target: str = "cohesive",
        E_atomic_by_index: Optional[torch.Tensor] = None,
        train: bool = True,
        show_batch_progress: bool = False,
        force_scale_unbiased: bool = False,
    ) -> Tuple[float, float, Dict[str, float]]:
        """
        Run one epoch over loader.

        Parameters
        ----------
        loader : DataLoader
            DataLoader for batches.
        optimizer : torch.optim.Optimizer or None
            Optimizer for training (None for validation).
        alpha : float
            Force loss weight (0.0 = energy only, 1.0 = forces only).
        energy_target : str
            Either 'cohesive' or 'total'.
        E_atomic_by_index : torch.Tensor, optional
            Atomic energies indexed by species, for cohesive conversion.
        train : bool
            Whether this is training (True) or validation (False).
        show_batch_progress : bool
            Whether to show per-batch progress bar.
        force_scale_unbiased : bool
            If True, apply optional sqrt(1/f) scaling to the per-batch
            force RMSE, where f is the supervised fraction of atoms
            with available force labels. This approximates constant
            loss magnitude when sub-sampling forces.

        Returns
        -------
        energy_rmse : float
            Energy RMSE for epoch.
        force_rmse : float
            Force RMSE for epoch (NaN if no forces).
        timing : dict
            Timing breakdown with keys: 'data_loading', 'loss_computation',
            'optimizer', 'total'.
        """
        energy_losses: List[float] = []
        force_losses: List[float] = []

        forward_time_total: float = 0.0
        backward_time_total: float = 0.0
        data_loading_time_total: float = 0.0
        loss_computation_time_total: float = 0.0
        optimizer_time_total: float = 0.0

        t_epoch_start = time.perf_counter()

        iterator = _iter_progress(
            loader,
            enable=show_batch_progress,
            desc=("train" if train else "val"),
        )
        t_batch_start = time.perf_counter()

        for batch in iterator:
            # Data loading time
            t_data_end = time.perf_counter()
            data_loading_time_total += t_data_end - t_batch_start

            # Energy view tensors
            features = batch["features"].to(self.device)
            species_indices = batch["species_indices"].to(self.device)
            n_atoms = batch["n_atoms"].to(self.device)
            energy_ref = batch["energy_ref"].to(self.device)

            # Ensure dtype consistency
            if self.dtype == torch.float64:
                features = features.double()
                energy_ref = energy_ref.double()
            else:
                features = features.float()
                energy_ref = energy_ref.float()

            # Convert targets to cohesive energies if configured
            if energy_target == "cohesive" and E_atomic_by_index is not None:
                per_atom_Ea = E_atomic_by_index[species_indices]
                batch_idx = torch.repeat_interleave(
                    torch.arange(len(n_atoms), device=self.device),
                    n_atoms.long(),
                )
                Ea_sum = torch.zeros(
                    len(n_atoms), dtype=energy_ref.dtype, device=self.device
                )
                Ea_sum.scatter_add_(0, batch_idx, per_atom_Ea)
                energy_ref = energy_ref - Ea_sum

            # Feature normalization
            features = self.normalizer.apply_feature_normalization(features)

            # Forward + loss computation
            t_forward_start = time.perf_counter()
            t_loss_start = time.perf_counter()

            # Energy loss
            E_shift = self.normalizer.E_shift
            E_scaling = self.normalizer.E_scaling

            energy_loss_t, _ = compute_energy_loss(
                features=features,
                energy_ref=energy_ref,
                n_atoms=n_atoms,
                network=self.model,
                species_indices=species_indices,
                E_shift=float(E_shift),
                E_scaling=float(E_scaling),
            )

            # Optional force loss
            force_loss_t: Optional[torch.Tensor] = None
            if alpha > 0.0 and batch["positions_f"] is not None:
                positions_f = batch["positions_f"].to(self.device)
                forces_ref_f = batch["forces_ref_f"].to(self.device)
                species_indices_f = batch["species_indices_f"].to(self.device)
                species_f = batch["species_f"]
                neighbor_info_f = batch["neighbor_info_f"]

                # dtype
                if self.dtype == torch.float64:
                    positions_f = positions_f.double()
                    forces_ref_f = forces_ref_f.double()
                else:
                    positions_f = positions_f.float()
                    forces_ref_f = forces_ref_f.float()

                force_loss_t, _ = compute_force_loss(
                    positions=positions_f,
                    species=species_f,
                    forces_ref=forces_ref_f,
                    descriptor=self.descriptor,
                    network=self.model,
                    species_indices=species_indices_f,
                    cell=None,
                    pbc=None,
                    E_scaling=float(E_scaling),
                    neighbor_info=neighbor_info_f,
                    chunk_size=None,
                    feature_mean=(
                        self.normalizer.feature_mean
                        if self.normalizer.normalize_features
                        else None
                    ),
                    feature_std=(
                        self.normalizer.feature_std
                        if self.normalizer.normalize_features
                        else None
                    ),
                    graph=batch.get("graph_f", None),
                    triplets=batch.get("triplets_f", None),
                    center_indices=None,
                )

                # Optional unbiased scaling of RMSE based on
                # supervised fraction
                if force_scale_unbiased:
                    try:
                        eff_total = int(batch.get("n_atoms_force_total", 0))
                        eff_supervised = int(
                            batch.get("n_atoms_force_supervised", 0))
                        if eff_total > 0:
                            eff_f = float(eff_supervised) / float(eff_total)
                            if eff_f > 0.0 and eff_f < 1.0:
                                scale = torch.tensor(
                                    eff_f,
                                    dtype=force_loss_t.dtype,
                                    device=force_loss_t.device,
                                )
                                # Approximate scaling for RMSE (MSE would
                                # not require scaling)
                                force_loss_t = force_loss_t / torch.sqrt(scale)
                    except Exception:
                        # If bookkeeping not present, skip scaling
                        pass

            # Combine losses
            if force_loss_t is None:
                combined = (1.0 - alpha) * energy_loss_t
            else:
                combined = (
                    (1.0 - alpha) * energy_loss_t + alpha * force_loss_t
                )

            t_loss_end = time.perf_counter()
            loss_computation_time_total += t_loss_end - t_loss_start
            forward_time_total += t_loss_end - t_forward_start

            # Backward + optimizer
            if train and optimizer is not None:
                t_backward_start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                combined.backward()
                t_optimizer_start = time.perf_counter()
                backward_time_total += t_optimizer_start - t_backward_start

                optimizer.step()
                t_optimizer_end = time.perf_counter()
                optimizer_time_total += t_optimizer_end - t_optimizer_start

            # Collect losses
            energy_losses.append(float(energy_loss_t.detach().cpu()))
            if force_loss_t is not None:
                force_losses.append(float(force_loss_t.detach().cpu()))

            # Prepare for next batch
            t_batch_start = time.perf_counter()

        t_epoch_end = time.perf_counter()

        # Compute RMSEs
        energy_rmse = float(
            sum(energy_losses) / max(1, len(energy_losses))
        )
        force_rmse = (
            float(sum(force_losses) / max(1, len(force_losses)))
            if force_losses
            else float("nan")
        )

        # Store timing for this epoch
        self.last_forward_time = forward_time_total
        self.last_backward_time = backward_time_total
        self.last_data_loading_time = data_loading_time_total
        self.last_loss_computation_time = loss_computation_time_total
        self.last_optimizer_time = optimizer_time_total

        timing = {
            "data_loading": data_loading_time_total,
            "loss_computation": loss_computation_time_total,
            "optimizer": optimizer_time_total,
            "total": t_epoch_end - t_epoch_start,
        }

        return energy_rmse, force_rmse, timing
