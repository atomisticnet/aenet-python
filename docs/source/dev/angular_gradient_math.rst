Analytical Gradient of Angular Symmetry Functions
==================================================

This page summarizes the mathematical formulas used by the analytical
angular gradients in ``ChebyshevDescriptor._compute_angular_gradients``.

Key definitions
---------------

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
-----

- Distances and unit vectors are evaluated with small epsilons for numerical
  stability (see analytical_gradients.rst).
- These expressions avoid autograd on geometric quantities and are used
  directly in the vectorized implementation.
