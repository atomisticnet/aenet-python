# Notebook Examples

This directory contains Jupyter notebook examples for `aenet-python`.

Available notebooks:

- `example-01-featurization.ipynb`: featurization with the Fortran-backed workflow
- `example-02-training.ipynb`: training ANN potentials with the Fortran-backed workflow
- `example-03-inference.ipynb`: inference with trained ANN potentials using the Fortran backend
- `example-04-torch-featurization.ipynb`: PyTorch-based featurization
- `example-05-torch-training.ipynb`: PyTorch-based training
- `example-06-torch-inference.ipynb`: inference with the PyTorch implementation
- `example-07-neighbor-list.ipynb`: neighbor-list usage with `TorchNeighborList`
- `example-08-libaenet-interface.ipynb`: direct `libaenet` and ASE-based workflows

Supporting files:

- `water.xyz`: small molecular example used by several notebooks
- `TiO2-cell.xsf`: TiO2 crystal example used by the `libaenet` notebook
- `pt-TiO2/`: PyTorch and ASCII TiO2 model files
- `nn-TiO2/`: ASCII neural-network potential files
- `xsf/` and `xsf-TiO2/`: example XSF structure datasets
