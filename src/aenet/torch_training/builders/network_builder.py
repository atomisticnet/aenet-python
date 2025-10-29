"""
Network architecture builder for PyTorch training.

Handles construction of per-species neural networks from architecture
specifications.
"""

import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class NetworkBuilder:
    """
    Builds neural network architectures for atomic energy prediction.

    Supports both aenet-PyTorch NetAtom (if available) and a fallback
    implementation using standard PyTorch modules.

    Parameters
    ----------
    descriptor : ChebyshevDescriptor
        Descriptor instance providing feature dimension and species info.
    device : torch.device
        Device for network.
    dtype : torch.dtype
        Data type for network parameters.
    """

    def __init__(self, descriptor, device: torch.device, dtype: torch.dtype):
        self.descriptor = descriptor
        self.device = device
        self.dtype = dtype

    @staticmethod
    def _import_netatom() -> Optional[type]:
        """
        Dynamically import aenet-PyTorch NetAtom from external/aenet-pytorch.

        Returns
        -------
        NetAtom class or None if not found.
        """
        try:
            # Determine project root (4 levels up from this file)
            root = Path(__file__).resolve().parents[4]
            net_path = (
                root / "external" / "aenet-pytorch" / "src" / "network.py"
            )
            if not net_path.exists():
                return None

            spec = importlib.util.spec_from_file_location(
                "aenet_pytorch.network", str(net_path)
            )
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)  # type: ignore

            if hasattr(module, "NetAtom"):
                return getattr(module, "NetAtom")
            return None
        except Exception:
            return None

    def validate_arch(
        self, arch: Dict[str, List[Tuple[int, str]]]
    ) -> Tuple[List[List[int]], List[List[str]]]:
        """
        Validate architecture and produce per-species hidden sizes and
        activations.

        Parameters
        ----------
        arch : dict
            {species_symbol: [(nodes, activation), ...]}
            Output layer is implicit.

        Returns
        -------
        hidden_sizes : list of list of int
            Per-species hidden layer sizes.
        activations : list of list of str
            Per-species activation functions.

        Raises
        ------
        ValueError
            On unsupported activation or missing species.
        """
        supported = {"linear", "tanh", "sigmoid"}
        species_order = list(self.descriptor.species)
        hidden_sizes: List[List[int]] = []
        activations: List[List[str]] = []

        for s in species_order:
            if s not in arch:
                raise ValueError(f"Species '{s}' missing in architecture.")
            layers = arch[s]
            hs: List[int] = []
            acts: List[str] = []
            for nodes, act in layers:
                act_l = act.lower()
                if act_l not in supported:
                    raise ValueError(
                        f"Unsupported activation '{act}' for species '{s}'. "
                        f"Supported: {sorted(supported)}"
                    )
                hs.append(int(nodes))
                acts.append(act_l)
            if len(hs) == 0:
                raise ValueError(
                    f"Architecture for species '{s}' must be non-empty."
                )
            hidden_sizes.append(hs)
            activations.append(acts)

        return hidden_sizes, activations

    def _build_fallback_per_species_mlps(
        self,
        n_features: int,
        species: List[str],
        hidden_sizes: List[List[int]],
        activations: List[List[str]],
    ) -> nn.ModuleList:
        """
        Fallback builder that mimics NetAtom.functions[iesp] layout.

        Returns
        -------
        nn.ModuleList
            Per-species nn.Sequential models mapping (F) -> (1).
        """
        act_map: Dict[str, nn.Module] = {
            "linear": nn.Identity(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        seqs: List[nn.Sequential] = []

        for i, _sp in enumerate(species):
            hs = hidden_sizes[i]
            acts = activations[i]
            layers: List[Tuple[str, nn.Module]] = []

            # First linear + act
            layers.append(
                (f"Linear_Sp{i+1}_F1", nn.Linear(n_features, hs[0]))
            )
            layers.append((f"Active_Sp{i+1}_F1", act_map[acts[0]]))

            # Hidden stacks
            for j in range(1, len(hs)):
                layers.append(
                    (f"Linear_Sp{i+1}_F{j+1}", nn.Linear(hs[j - 1], hs[j]))
                )
                layers.append((f"Active_Sp{i+1}_F{j+1}", act_map[acts[j]]))

            # Output layer
            layers.append(
                (f"Linear_Sp{i+1}_F{len(hs)+1}", nn.Linear(hs[-1], 1))
            )
            seqs.append(nn.Sequential(dict(layers)))  # type: ignore

        return nn.ModuleList(seqs)

    def build_network(
        self, arch: Dict[str, List[Tuple[int, str]]]
    ) -> nn.Module:
        """
        Build NetAtom (preferred) or fallback per-species MLPs.

        Parameters
        ----------
        arch : dict
            Architecture specification per species.

        Returns
        -------
        nn.Module
            Network with attributes:
            - .functions: ModuleList of per-species Sequential MLPs
            - .device: device string
        """
        species = list(self.descriptor.species)
        n_features = int(self.descriptor.get_n_features())
        hidden_sizes, activations = self.validate_arch(arch)

        NetAtom = self._import_netatom()

        if NetAtom is not None:
            # NetAtom expects lists per species
            input_size = [n_features for _ in species]
            alpha = 1.0
            net = NetAtom(
                input_size=input_size,
                hidden_size=hidden_sizes,
                species=species,
                activations=activations,
                alpha=alpha,
                device=str(self.device),
            )
            # Ensure dtype/device
            if self.dtype == torch.float64:
                net = net.double()
            else:
                net = net.float()
            # NetAtom stores device string
            net.device = str(self.device)
            net.to(self.device)
            return net

        # Fallback: simple wrapper with .functions and .device
        class _FallbackNet(nn.Module):
            def __init__(self_inner):
                super().__init__()
                self_inner.functions = (
                    self._build_fallback_per_species_mlps(
                        n_features=n_features,
                        species=species,
                        hidden_sizes=hidden_sizes,
                        activations=activations,
                    )
                )
                self_inner.species = species  # for debugging
                self_inner.device = str(self.device)

            def forward(self_inner, *args, **kwargs):
                """Not used; adapter calls .functions."""
                raise RuntimeError(
                    "Use EnergyModelAdapter to call per-atom energies."
                )

        net = _FallbackNet()
        if self.dtype == torch.float64:
            net = net.double()
        else:
            net = net.float()
        net.to(self.device)
        return net
