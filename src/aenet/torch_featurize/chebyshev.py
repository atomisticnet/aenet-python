"""
Vectorized Chebyshev Polynomial Evaluation for PyTorch.

This module implements Chebyshev polynomial evaluation using the explicit
cosine form T_n(x) = cos(n * arccos(x)) instead of recurrence relations,
enabling efficient vectorization and GPU acceleration.

References
----------
    N. Artrith, A. Urban, and G. Ceder, PRB 96 (2017) 014112
"""

from typing import Tuple

import torch
import torch.nn as nn


class ChebyshevPolynomials(nn.Module):
    """
    Vectorized Chebyshev polynomial evaluation using cosine form.

    Uses T_n(x) = cos(n * arccos(x)) for numerical stability and
    efficient vectorization across all polynomial orders.

    Parameters
    ----------
    max_order : int
        Maximum Chebyshev polynomial order to compute
    r_min : float
        Minimum distance (inner cutoff) in Angstroms
    r_max : float
        Maximum distance (outer cutoff) in Angstroms
    dtype : torch.dtype, optional
        Data type for computations (default: torch.float64)

    Examples
    --------
    >>> cheb = ChebyshevPolynomials(max_order=5, r_min=0.5, r_max=4.0)
    >>> r = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    >>> T = cheb(r)  # Shape: (3, 6) for orders 0-5
    """

    def __init__(
        self,
        max_order: int,
        r_min: float,
        r_max: float,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.max_order = max_order
        self.r_min = r_min
        self.r_max = r_max
        self.dtype = dtype

        # Register order indices as buffer (for automatic device transfer)
        orders = torch.arange(max_order + 1, dtype=dtype)
        self.register_buffer("orders", orders)

    def rescale_distances(self, r: torch.Tensor) -> torch.Tensor:
        """
        Rescale distances from [r_min, r_max] to [-1, 1].

        The Chebyshev polynomials are defined on [-1, 1], so distances
        must be rescaled using:
            x = (2*r - r_min - r_max) / (r_max - r_min)

        Parameters
        ----------
        r : torch.Tensor
            Distances in Angstroms, any shape

        Returns
        -------
        torch.Tensor
            Rescaled distances in [-1, 1], same shape as r

        Notes
        -----
        Values are clamped to [-1, 1] for numerical stability.
        """
        x = (2.0 * r - self.r_min - self.r_max) / (self.r_max - self.r_min)
        # Clamp to valid range for numerical stability
        x = torch.clamp(x, -1.0, 1.0)
        return x

    def cutoff_function(self, r: torch.Tensor, Rc: float) -> torch.Tensor:
        """
        Cosine cutoff function.

        Implements: fc(r) = 0.5 * [cos(π*r/Rc) + 1] for r < Rc
                    fc(r) = 0                       for r >= Rc

        Parameters
        ----------
        r : torch.Tensor
            Distances, any shape
        Rc : float
            Cutoff radius in Angstroms

        Returns
        -------
        torch.Tensor
            Cutoff function values, same shape as r

        Notes
        -----
        The cutoff function smoothly goes to zero at r=Rc, ensuring
        continuous features with continuous first derivatives.
        """
        fc = torch.where(
            r < Rc,
            0.5 * (torch.cos(torch.pi * r / Rc) + 1.0),
            torch.zeros_like(r),
        )
        return fc

    def cutoff_derivative(self, r: torch.Tensor, Rc: float) -> torch.Tensor:
        """
        Derivative of the cosine cutoff function.

        Implements: dfc/dr = -0.5 * π/Rc * sin(π*r/Rc) for r < Rc
                    dfc/dr = 0                          for r >= Rc

        Parameters
        ----------
        r : torch.Tensor
            Distances, any shape
        Rc : float
            Cutoff radius in Angstroms

        Returns
        -------
        torch.Tensor
            Derivative of cutoff function, same shape as r
        """
        dfc = torch.where(
            r < Rc,
            -0.5 * torch.pi / Rc * torch.sin(torch.pi * r / Rc),
            torch.zeros_like(r),
        )
        return dfc

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Chebyshev polynomials for given distances.

        Uses the explicit formula T_n(x) = cos(n * arccos(x)) which
        allows computing all orders simultaneously.

        Parameters
        ----------
        r : torch.Tensor
            Distances in Angstroms, shape (..., N)

        Returns
        -------
        torch.Tensor
            Chebyshev polynomials, shape (..., N, max_order+1)
            T[..., i, n] = T_n(x_i) where x_i is rescaled r[..., i]

        Notes
        -----
        Broadcasting:
            - arccos_x has shape (..., N)
            - self.orders has shape (max_order+1,)
            - Result has shape (..., N, max_order+1)
        """
        # Rescale distances to [-1, 1]
        x = self.rescale_distances(r)

        # Compute arccos once (clamping for numerical stability)
        arccos_x = torch.arccos(x.clamp(-1.0, 1.0))

        # Compute T_n(x) = cos(n * arccos(x)) for all n simultaneously
        # Broadcasting: arccos_x[..., N, 1] * orders[max_order+1]
        T = torch.cos(self.orders * arccos_x.unsqueeze(-1))

        return T

    def evaluate_with_derivatives(
        self, r: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate Chebyshev polynomials and their derivatives.

        Uses the relationship:
            dT_n/dx = n * U_{n-1}(x)
        where U_n are Chebyshev polynomials of the second kind:
            U_n(x) = sin((n+1)*arccos(x)) / sqrt(1-x²)

        Parameters
        ----------
        r : torch.Tensor
            Distances in Angstroms, shape (..., N)

        Returns
        -------
        T : torch.Tensor
            Chebyshev polynomials, shape (..., N, max_order+1)
        dT_dr : torch.Tensor
            Derivatives w.r.t. r, shape (..., N, max_order+1)

        Notes
        -----
        The derivative uses the chain rule:
            dT_n/dr = dT_n/dx * dx/dr
        where dx/dr = 2/(r_max - r_min) from the rescaling.
        """
        # Rescale distances to [-1, 1]
        x = self.rescale_distances(r)

        # Compute arccos (clamped for stability)
        arccos_x = torch.arccos(x.clamp(-1.0, 1.0))

        # Chebyshev polynomials T_n(x) = cos(n * arccos(x))
        T = torch.cos(self.orders * arccos_x.unsqueeze(-1))

        # For derivatives, we need U_{n-1}(x)
        # U_n(x) = sin((n+1)*arccos(x)) / sqrt(1-x²)

        # Compute sqrt term with small epsilon for numerical stability
        sqrt_term = torch.sqrt((1.0 - x**2).clamp(min=1e-10))

        # Compute U polynomials for all orders
        # U_n corresponds to order n, so U_{n-1} is at index n-1
        U_orders = torch.arange(
            self.max_order + 1, dtype=self.dtype, device=r.device
        )
        U = torch.sin(
            (U_orders + 1) * arccos_x.unsqueeze(-1)
        ) / sqrt_term.unsqueeze(-1)

        # dT_n/dx = n * U_{n-1}
        # For n=0: derivative is 0
        # For n≥1: use n * U[n-1]
        dT_dx = torch.zeros_like(T)
        if self.max_order >= 1:
            dT_dx[..., 1:] = self.orders[1:] * U[..., :-1]

        # Chain rule: dT/dr = dT/dx * dx/dr
        dx_dr = 2.0 / (self.r_max - self.r_min)
        dT_dr = dT_dx * dx_dr

        return T, dT_dr


class RadialBasis(nn.Module):
    """
    Radial basis functions combining Chebyshev polynomials with cutoff.

    Implements: G_rad = T_n(r) * fc(r)

    Parameters
    ----------
    rad_order : int
        Maximum order for radial Chebyshev polynomials
    rad_cutoff : float
        Radial cutoff radius in Angstroms
    min_cutoff : float, optional
        Minimum cutoff (inner radius) in Angstroms (default: 0.55)
    dtype : torch.dtype, optional
        Data type for computations (default: torch.float64)

    Examples
    --------
    >>> rad_basis = RadialBasis(rad_order=10, rad_cutoff=4.0)
    >>> distances = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    >>> G_rad = rad_basis(distances)  # Shape: (3, 11) for orders 0-10
    """

    def __init__(
        self,
        rad_order: int,
        rad_cutoff: float,
        min_cutoff: float = 0.55,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()

        # Chebyshev polynomials are evaluated on [0, Rc], not [min_cutoff, Rc]
        # min_cutoff is only used for neighbor filtering, not polynomial domain
        self.cheb = ChebyshevPolynomials(
            max_order=rad_order,
            r_min=0.0,  # Always 0.0 for radial basis
            r_max=rad_cutoff,
            dtype=dtype,
        )
        self.rad_cutoff = rad_cutoff
        self.rad_order = rad_order

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Evaluate radial symmetry functions.

        Parameters
        ----------
        distances : torch.Tensor
            Pairwise distances in Angstroms, shape (num_pairs,)

        Returns
        -------
        torch.Tensor
            Radial features, shape (num_pairs, rad_order+1)
        """
        # Chebyshev polynomials
        T = self.cheb(distances)  # (num_pairs, rad_order+1)

        # Cutoff function
        fc = self.cheb.cutoff_function(distances, self.rad_cutoff)

        # Combine: G_rad = T * fc
        G_rad = T * fc.unsqueeze(-1)

        return G_rad

    def forward_with_derivatives(
        self, distances: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate radial symmetry functions with derivatives.

        Uses the product rule:
            d(T*fc)/dr = dT/dr * fc + T * dfc/dr

        Parameters
        ----------
        distances : torch.Tensor
            Pairwise distances in Angstroms, shape (num_pairs,)

        Returns
        -------
        G_rad : torch.Tensor
            Radial features, shape (num_pairs, rad_order+1)
        dG_rad_dr : torch.Tensor
            Derivatives w.r.t. distance, shape (num_pairs, rad_order+1)
        """
        # Polynomials and derivatives
        T, dT_dr = self.cheb.evaluate_with_derivatives(distances)

        # Cutoff and its derivative
        fc = self.cheb.cutoff_function(distances, self.rad_cutoff)
        dfc_dr = self.cheb.cutoff_derivative(distances, self.rad_cutoff)

        # Product rule: d(T*fc)/dr = dT/dr * fc + T * dfc/dr
        G_rad = T * fc.unsqueeze(-1)
        dG_rad_dr = dT_dr * fc.unsqueeze(-1) + T * dfc_dr.unsqueeze(-1)

        return G_rad, dG_rad_dr


class AngularBasis(nn.Module):
    """
    Angular basis functions using Chebyshev polynomials.

    For a triplet of atoms (i, j, k), computes:
        G_ang = T_n(cos θ_ijk) * fc(r_ij) * fc(r_ik)
    where θ_ijk is the angle at atom i.

    Parameters
    ----------
    ang_order : int
        Maximum order for angular Chebyshev polynomials
    ang_cutoff : float
        Angular cutoff radius in Angstroms
    min_cutoff : float, optional
        Minimum cutoff (not used for angular, kept for consistency)
    dtype : torch.dtype, optional
        Data type for computations (default: torch.float64)

    Notes
    -----
    For angular features, cos(θ) is already in [-1, 1], so we use
    r_min=-1.0 and r_max=1.0 (no rescaling needed).
    """

    def __init__(
        self,
        ang_order: int,
        ang_cutoff: float,
        min_cutoff: float = 0.55,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()

        # For cos(θ), already in [-1, 1], so no rescaling needed
        self.cheb = ChebyshevPolynomials(
            max_order=ang_order, r_min=-1.0, r_max=1.0, dtype=dtype
        )
        self.ang_cutoff = ang_cutoff
        self.ang_order = ang_order

    def forward(
        self, r_ij: torch.Tensor, r_ik: torch.Tensor, cos_theta: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate angular symmetry functions.

        Parameters
        ----------
        r_ij : torch.Tensor
            Distances from atom i to atom j, shape (num_triplets,)
        r_ik : torch.Tensor
            Distances from atom i to atom k, shape (num_triplets,)
        cos_theta : torch.Tensor
            Cosine of angles θ_ijk, shape (num_triplets,)

        Returns
        -------
        torch.Tensor
            Angular features, shape (num_triplets, ang_order+1)

        Notes
        -----
        The cosine of angles is clamped to [-1, 1] for numerical stability.
        """
        # Chebyshev of cos(θ) - no rescaling needed as already in [-1,1]
        # Since r_min=-1, r_max=1, rescaling is identity: x = cos_theta
        T_theta = self.cheb(cos_theta)

        # Cutoff functions for both distances
        fc_ij = self.cheb.cutoff_function(r_ij, self.ang_cutoff)
        fc_ik = self.cheb.cutoff_function(r_ik, self.ang_cutoff)

        # Combine: G_ang = T(cos θ) * fc(r_ij) * fc(r_ik)
        G_ang = T_theta * (fc_ij * fc_ik).unsqueeze(-1)

        return G_ang
