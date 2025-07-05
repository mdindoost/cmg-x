# Contributing to CMG-X

Welcome! We're excited that you're interested in contributing to **CMG-X**, a multiscale graph coarsening library built on Combinatorial Multigrid (CMG). Whether you're fixing bugs, improving performance, writing documentation, or adding new features — your contributions are highly valued.

---

## 🧱 How to Contribute

### 1. Fork and Clone
First, fork the repository and clone it to your local machine:
```bash
git clone https://https://github.com/mdindoost/cmg-x
cd cmg-x
```

### 2. Set Up Your Environment
Install the project in development mode:
```bash
pip install -e .
```

Build the C extension if not already:
```bash
gcc -shared -fPIC -O3 src/cmgCluster.c -o cmgx/libcmgCluster.so
```

### 3. Run Tests
Always verify changes with:
```bash
pytest tests/
```

---

## 📂 Directory Structure

```
cmg-x/
├── cmgx/               # Core Python + C interface
├── src/                # C source code
├── include/            # Header files
├── tests/              # Unit tests
├── examples/           # Demos and benchmarks
├── docs/               # Optional documentation
├── setup.py            # Packaging setup
└── README.md           # Project overview
```

---

## 🧪 Contribution Types

- 💡 New pooling methods or CMG variants
- 🚀 Performance improvements or CUDA support
- 🧠 Research: ablations, benchmarks, visualizations
- 📖 Documentation: tutorials, docstrings, API guides
- 🧪 Tests: Add missing coverage or robustness checks

---

## 🧼 Code Guidelines

- Follow PEP8 formatting (`black` recommended)
- Keep functions modular and documented
- Use clear variable names (e.g. `L`, `P`, `Xc`)
- Match PyTorch/PyG idioms wherever possible

---

## 📬 Submitting a Pull Request

1. Create a branch: `git checkout -b feature/your-feature-name`
2. Commit your changes with a clear message
3. Push to your fork and open a pull request

Please include:
- What your PR does
- Any related issues or references
- Before/after benchmarks if performance-related

---

## 🤝 Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/). All interactions should be respectful, inclusive, and collaborative.

---

Thank you for helping improve CMG-X!
