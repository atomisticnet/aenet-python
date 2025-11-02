Analytical Gradients for Chebyshev AUC Descriptors
====================================================

Overview
--------

This document summarizes the semi-analytical, vectorized gradient implementation
for the AUC (Artrith-Urban-Ceder) Chebyshev descriptor in ``aenet-python``.
It explains the design goals, numerical stability measures, and the code
structure that computes gradients of both radial and angular symmetry functions
with respect to atomic positions.

Motivation
~~~~~~~~~~

- Performance: Avoid the slow feature-by-feature autograd loop by vectorizing
  across all pairs and triplets.
- Robustness: Prevent NaNs arising from zero-distance image interactions in
  periodic systems by avoiding differentiating through norms in autograd.

Key Interfaces
--------------

- Module: ``src/aenet/torch_featurize/featurize.py``
- Class: ``ChebyshevDescriptor``
- Feature computation entry points:
  - ``forward_from_positions(positions, species, cell, pbc)``
  - ``forward(positions, species, neighbor_indices, neighbor_vectors)``
- Gradient computation:

  - ``compute_feature_gradients(positions, species, cell, pbc)`` returns ``(features, gradients)`` with shape ``features: (N, F)``, ``gradients: (N, F, N, 3)``
  - Internal methods:
    - ``_compute_radial_gradients(...)``
    - ``_compute_angular_gradients(...)``

Numerical Stability
-------------------

- Distances are always computed with small epsilons to avoid singularities:
  - ``d = sqrt(r · r + eps_dist)`` with ``eps_dist = 1e-20``
  - Unit vectors normalized as ``u = r / (|r| + eps_norm)`` with ``eps_norm = 1e-12``
- Angular cosine clamped to the valid range:
  - ``cos_theta = dot(u_ij, u_ik).clamp(-1, 1)``
- No autograd is used for the geometric derivatives (e.g., w.r.t. coordinates);
  instead, closed-form expressions are applied (see below).
- Cutoff functions are smooth (cosine cutoff), and their derivatives are used
  analytically.

Radial Gradients (Summary)
--------------------------

For each neighbor pair ``(i, j)`` with displacement ``r_ij`` and distance
``d_ij = ||r_ij||``:

- Basis and derivative from ``RadialBasis``:
  - ``G_rad(d_ij)`` and ``dG_rad/dd``
- Chain rule to coordinates:
  - ``dG/dr_ij_vec = (dG/dd) * (r_ij / |r_ij|)``

Contributions are accumulated to atom ``i`` (negative sign) and atom ``j``
(positive sign) for both unweighted and typespin-weighted features. The
accumulation uses vectorized ``index_add_`` into a flattened center/target
index to achieve good performance.

Angular Gradients (Summary)
---------------------------

For each neighbor triplet ``(i, j, k)`` with displacements ``r_ij, r_ik``:

- Distances and unit vectors:
  - ``d_ij = ||r_ij||``, ``d_ik = ||r_ik||``
  - ``u_ij = r_ij / (|r_ij| + eps_norm)``, etc.
- Cosine of the angle at ``i``:
  - ``cos_theta = dot(u_ij, u_ik)`` (clamped to ``[-1, 1]``)
- Basis partial derivatives from ``AngularBasis``:
  - ``dG/dcos``, ``dG/dr_ij``, ``dG/dr_ik``

Geometric derivatives (see detailed derivation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \frac{\partial \cos\theta}{\partial \mathbf{r}_j}
   = \frac{1}{r_{ij}}\left(\frac{\mathbf{r}_{ik}}{r_{ik}}
     - \cos\theta\,\frac{\mathbf{r}_{ij}}{r_{ij}}\right)

.. math::

   \frac{\partial \cos\theta}{\partial \mathbf{r}_k}
   = \frac{1}{r_{ik}}\left(\frac{\mathbf{r}_{ij}}{r_{ij}}
     - \cos\theta\,\frac{\mathbf{r}_{ik}}{r_{ik}}\right)

.. math::

   \frac{\partial \cos\theta}{\partial \mathbf{r}_i}
   = -\left(\frac{\partial \cos\theta}{\partial \mathbf{r}_j}
     + \frac{\partial \cos\theta}{\partial \mathbf{r}_k}\right)

Final chain rule to coordinates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \frac{\mathrm{d} G}{\mathrm{d}\mathbf{r}_j}
   = \frac{\partial G}{\partial \cos\theta}\,
     \frac{\mathrm{d}\cos\theta}{\mathrm{d}\mathbf{r}_j}
     + \frac{\partial G}{\partial r_{ij}}\,
       \frac{\mathbf{r}_{ij}}{r_{ij}}

.. math::

   \frac{\mathrm{d} G}{\mathrm{d}\mathbf{r}_k}
   = \frac{\partial G}{\partial \cos\theta}\,
     \frac{\mathrm{d}\cos\theta}{\mathrm{d}\mathbf{r}_k}
     + \frac{\partial G}{\partial r_{ik}}\,
       \frac{\mathbf{r}_{ik}}{r_{ik}}

.. math::

   \frac{\mathrm{d} G}{\mathrm{d}\mathbf{r}_i}
   = \frac{\partial G}{\partial \cos\theta}\,
     \frac{\mathrm{d}\cos\theta}{\mathrm{d}\mathbf{r}_i}
     - \frac{\partial G}{\partial r_{ij}}\,
       \frac{\mathbf{r}_{ij}}{r_{ij}}
     - \frac{\partial G}{\partial r_{ik}}\,
       \frac{\mathbf{r}_{ik}}{r_{ik}}

Implementation Notes
--------------------

- Code: ``ChebyshevDescriptor._compute_angular_gradients``
  - Vectorized derivation for all triplets in a structure.
  - Avoids ``torch.autograd.grad`` on geometric terms; only uses analytical expressions and partials provided by ``AngularBasis.forward_with_derivatives``.
  - Accumulation for unweighted and typespin-weighted angular features is done via flattened ``index_add_`` into an ``(n_atoms * n_atoms, 3)`` workspace per feature index, reshaped back to ``(n_atoms, n_atoms, 3)``.

- Feature ordering (consistent with Fortran):
  - Single species: ``[radial, angular]``
  - Multi-species: ``[rad_unweighted, ang_unweighted, rad_weighted, ang_weighted]``

Testing and Validation
----------------------

- Periodic gradient tests in
  ``src/aenet/torch_featurize/tests/test_gradients.py::TestPeriodicGradients``
- NaN reproduction check:
  ``reproduce_nan.py`` (verifies that no NaNs occur under PBC self-interactions)
- Stress tests:
  ``test_stress_gradients.py`` (small cells, dense neighbor environments,
  multi-species code paths)
- Finite-difference spot checks to confirm analytical correctness.

Performance Considerations
--------------------------

- The radial part is fully vectorized.
- The angular accumulation currently loops over ``n_ang`` for index-add
  efficiency and clarity; if profiling identifies it as a bottleneck, a
  batched scatter approach can be implemented to remove this small loop.

Analytical Gradient of Angular Symmetry Functions
-------------------------------------------------

Key definitions
~~~~~~~~~~~~~~~

- r_ij = ||r_j - r_i||, r_ik = ||r_k - r_i||
- u_ij = (r_j - r_i) / r_ij, u_ik = (r_k - r_i) / r_ik
- cos(theta) = u_ij · u_ik (clamped to [-1, 1])

Angular symmetry function:

  G_ang = T_n(cos(theta)) * f_c(r_ij) * f_c(r_ik)

Partials from basis:

- dG/d(cos) = dT_n/d(cos(theta)) * f_c(r_ij) * f_c(r_ik)
- dG/dr_ij = T_n(cos(theta)) * df_c(r_ij)/dr_ij * f_c(r_ik)
- dG/dr_ik = T_n(cos(theta)) * f_c(r_ij) * df_c(r_ik)/dr_ik

Geometric derivatives:

- dcos/dr_j = (1/r_ij) * (u_ik - cos(theta) * u_ij)
- dcos/dr_k = (1/r_ik) * (u_ij - cos(theta) * u_ik)
- dcos/dr_i = - (dcos/dr_j + dcos/dr_k)

Final gradients:

- dG/dr_j = (dG/dcos) * (dcos/dr_j) + (dG/dr_ij) * u_ij
- dG/dr_k = (dG/dcos) * (dcos/dr_k) + (dG/dr_ik) * u_ik
- dG/dr_i = (dG/dcos) * (dcos/dr_i) - (dG/dr_ij) * u_ij - (dG/dr_ik) * u_ik

Notes
~~~~~

- Distances and unit vectors are evaluated with small epsilons for numerical
  stability (see analytical_gradients.rst).
- These expressions avoid autograd on geometric quantities and are used
  directly in the vectorized implementation.
