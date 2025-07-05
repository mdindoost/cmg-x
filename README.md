# CMG-X: A General Multiscale Graph Coarsening Library

CMG-X is a modular, efficient, and fully differentiable library for **multiscale graph coarsening**, inspired by the Combinatorial Multigrid (CMG) algorithm. It supports pooling, unpooling, and interpolation for GNNs, vector DBs, and image superpixel graphs.

---

## 🚀 Quick Start

```bash
# Clone and install (editable mode)
git clone https://github.com/mdindoost/cmg-x.git
cd cmg-x
pip install -e .
```

Run example:
```bash
python examples/train_autoencoder.py
```

---

## 🔧 Installation

Requirements:
- Python ≥ 3.8
- PyTorch ≥ 1.13
- torch_geometric
- scipy, numpy

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install -r requirements.txt
```

---

## 🧠 Features

- `cmg_pool(X, L)`: Spectral coarsening of node features and Laplacian.
- `cmg_unpool(X_coarse, P)`: Coarse-to-fine interpolation.
- `cmg_interpolate_multilevel`: Multilevel feature upsampling.
- PyG wrappers: `CMGPooling`, `CMGUnpooling` (fully differentiable).
- Multilevel hierarchy construction and analysis.

---

## 🧪 Testing

```bash
pytest tests/
```

Includes:
- ✅ Pooling / unpooling round-trip tests
- ✅ Autoencoder reconstruction MSE
- ✅ Jacobian gradient checks
- ✅ Multilevel interpolation

---

## 📊 Benchmarks

Run:
```bash
python examples/benchmark_pooling.py
```

Supports:
- Cora, Citeseer, Pubmed
- CMG vs. TopKPool, SAGPool, ASAP, DiffPool
- Output: Accuracy, compression, runtime

---

## 📎 Citation (Coming Soon)

If you use this library in your research, please cite:

```
@article{md2025cmgx,
  title={CMG-X: A General Multiscale Graph Coarsening Library for GNNs and Graph Databases},
  author={Dindoost, Mohammad and Collaborators},
  year={2025},
  journal={arXiv preprint arXiv:25xx.xxxxx}
}
```

---

## 🧩 Maintainer

Mohammad Dindoost (md724@NJIT.edu)

For questions, contact via GitHub or open an issue.
