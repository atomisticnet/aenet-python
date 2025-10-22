#!/usr/bin/env python
"""
Basic test to verify gradient computation works through featurization.
"""

import torch
from src.aenet.torch_featurize.featurize import ChebyshevDescriptor

# Create water molecule
positions = torch.tensor([
    [0.0, 0.0, 0.0],      # O
    [0.96, 0.0, 0.0],     # H1
    [-0.24, 0.93, 0.0]    # H2
], dtype=torch.float64, requires_grad=True)

species = ['O', 'H', 'H']

# Create descriptor
descriptor = ChebyshevDescriptor(
    species=['O', 'H'],
    rad_order=10,
    rad_cutoff=4.0,
    ang_order=3,
    ang_cutoff=1.5,
    device='cpu'
)

# Compute features
features = descriptor(positions, species)
print(f"Features shape: {features.shape}")
print(f"Features require grad: {features.requires_grad}")

# Compute gradient of sum of features w.r.t. positions
loss = features.sum()
loss.backward()

print(f"\nPositions gradient shape: {positions.grad.shape}")
print(f"Positions gradient non-zero: {torch.any(positions.grad != 0)}")
print(f"Gradient values:\n{positions.grad}")

# Test compute_feature_gradients method
positions2 = torch.tensor([
    [0.0, 0.0, 0.0],
    [0.96, 0.0, 0.0],
    [-0.24, 0.93, 0.0]
], dtype=torch.float64)

features2, gradients = descriptor.compute_feature_gradients(positions2, species)
print(f"\nFull gradient tensor shape: {gradients.shape}")
print(f"Gradient tensor non-zero: {torch.any(gradients != 0)}")

print("\n✓ Gradient computation successful!")
