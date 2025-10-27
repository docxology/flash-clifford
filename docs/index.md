# Flash Clifford Documentation

`flash-clifford` provides efficient Triton-based implementations of Clifford algebra-based models for neural networks. This library implements hardware-accelerated geometric operations that achieve state-of-the-art performance on equivariant learning tasks.

## Overview

Clifford algebras provide a natural framework for geometric deep learning by representing data in terms of multivectors - mathematical objects that encode scalar, vector, bivector, and higher-grade components corresponding to irreducible representations of the orthogonal group O(n). The geometric product, a fundamental operation in Clifford algebra, enables efficient computation of equivariant transformations.

This implementation focuses on optimizing the performance bottlenecks in Clifford neural networks through:

- **Triton-accelerated kernels** for geometric operations
- **Kernel fusion** combining activation, geometric product, and normalization
- **Optimized memory layouts** for efficient tensor operations
- **Comprehensive benchmarking** against baseline implementations

## Technical Foundation

### Mathematical Structure

The library implements Clifford algebras Cl(n,0) for Euclidean spaces:

- **Cl(2,0)**: 4-dimensional multivector space (1 scalar + 2 vectors + 1 pseudoscalar)
- **Cl(3,0)**: 8-dimensional multivector space (1 scalar + 3 vectors + 3 bivectors + 1 pseudoscalar)

Each multivector component corresponds to irreducible representations of O(n):
- Scalars → trivial representation (0e)
- Vectors → standard representation (1o)
- Bivectors → adjoint representation (1e)
- Higher grades → higher-dimensional representations

### Implementation Architecture

The library is structured around several key components:

1. **Triton Kernels** (`ops/`): Hand-optimized CUDA kernels for geometric operations
2. **Neural Network Layers** (`modules/`): PyTorch modules implementing Clifford layers
3. **Baseline Implementations** (`modules/baseline.py`): Reference implementations for correctness
4. **Comprehensive Testing** (`tests/`): Verification, benchmarking, and performance analysis

### Performance Characteristics

Achieved improvements over baseline implementations:
- **Runtime**: 2-5× speedup through kernel optimization and fusion
- **Memory**: 30-50% reduction in peak memory usage
- **Scalability**: Linear scaling with batch size and feature dimensions
- **Numerical accuracy**: Maintains correctness within 1e-5 tolerance

## Quick Start

```python
import torch
from modules.layer import Layer

# Input: multivectors in 3D of shape (8, batch, features)
x = torch.randn(8, 4096, 512).cuda()

# Linear layer: grade-wise linear + weighted GP
layer = Layer(512, dims=3, normalize=True, use_fc=False).cuda()

output = layer(x)
```

## Documentation Structure

### [Core Concepts](core-concepts.md)
Mathematical foundations of Clifford algebra, multivectors, and the geometric product.

### [Operations API](operations.md)
Detailed documentation of fused operations and geometric product implementations.

### [Architecture](architecture.md)
Neural network layer implementations and design patterns.

### [Performance Guide](performance.md)
Performance optimizations, benchmarking methodology, and results analysis.

### [Implementation Details](implementation.md)
Triton kernel implementations and low-level optimizations.

### [API Reference](api-reference.md)
Complete API documentation for all modules and functions.

### [Usage Examples](examples.md)
Practical examples and tutorials for common use cases.

## Mathematical Foundation

Clifford algebra Cl(p,q,r) extends vector algebra with a product that generalizes the dot and cross products. Elements are **multivectors** - linear combinations of basis elements:

```
A = a₀ + a₁e₁ + a₂e₂ + a₃e₃ + a₁₂e₁₂ + a₁₃e₁₃ + a₂₃e₂₃ + a₁₂₃e₁₂₃
```

The **geometric product** AB combines multivectors while preserving equivariance properties essential for geometric deep learning.

## Performance Highlights

Flash Clifford achieves significant performance improvements over baseline implementations:

- **Runtime speedup**: 2-5× faster than PyTorch baselines
- **Memory efficiency**: 30-50% reduction in peak memory usage
- **Scalability**: Linear scaling with batch size and feature dimensions

These optimizations are particularly beneficial for large-scale geometric learning tasks in computer vision, physics simulation, and molecular modeling.

## Citation

If you find this work helpful, please cite:

```bibtex
@software{flashclifford2025,
  title  = {Flash Clifford: Hardware-Efficient Implementation of Clifford Algebra Neural Networks},
  author = {Zhdanov, Maksim},
  url    = {https://github.com/maxxxzdn/flash-clifford},
  year   = {2025}
}
```

## Installation

### Basic Installation

```bash
# From source
git clone https://github.com/maxxxzdn/flash-clifford.git
cd flash-clifford
pip install -e .

# Development installation with tests
pip install -e ".[dev]"
```

### Requirements

**Core Requirements:**
- PyTorch >= 2.0
- Triton >= 3.0
- CUDA-capable GPU (compute capability 7.0+ for optimal performance)

**Development Requirements:**
- NumPy
- matplotlib
- pytest (for testing)
- black, isort, flake8 (for code formatting)

### Hardware Support

| GPU Architecture | Compute Capability | Performance Level | Notes |
|-----------------|-------------------|-------------------|-------|
| RTX 40-series   | 8.9               | Excellent         | Optimal performance |
| RTX 30-series   | 8.6               | Excellent         | Full feature support |
| RTX 20-series   | 7.5               | Good              | Baseline performance |
| A100/H100       | 8.0               | Excellent         | Server-grade performance |
| V100            | 7.0               | Acceptable        | Minimum requirement |

## Development

### Project Structure

```
flash-clifford/
├── modules/           # Neural network layers
│   ├── layer.py      # Optimized layer implementations
│   └── baseline.py   # Reference implementations
├── ops/              # Triton kernel operations
│   ├── p2m0.py       # 2D geometric product kernels
│   ├── p3m0.py       # 3D geometric product kernels
│   ├── fc_p2m0.py    # 2D fully connected kernels
│   └── fc_p3m0.py    # 3D fully connected kernels
├── tests/            # Comprehensive test suite
│   ├── benchmarks/   # Performance benchmarks
│   ├── utils.py      # Testing utilities
│   └── baselines.py  # Baseline implementations
└── docs/             # This documentation
```

### Development Workflow

1. **Code Style**: Follow PEP 8 with black formatting
2. **Testing**: All changes must pass existing tests
3. **Benchmarking**: Performance regressions must be justified
4. **Documentation**: Update docs alongside code changes

```bash
# Run tests
python -m pytest tests/

# Run benchmarks
python -m tests.benchmarks.layer_3d

# Format code
black . && isort .

# Check linting
flake8 modules/ ops/ tests/
```

## Research Context

### Related Work

Flash Clifford builds upon several key research contributions:

1. **Clifford Neural Networks**: [Ruhe et al., 2023](https://arxiv.org/abs/2305.11141) - Original CEGNN formulation
2. **Equivariant Networks**: [Weiler et al., 2018](https://arxiv.org/abs/1802.08219) - E(n) equivariant CNNs
3. **Geometric Deep Learning**: [Bronstein et al., 2021](https://arxiv.org/abs/2104.13478) - Comprehensive survey

### Applications

The library has been validated on several benchmark tasks:

- **N-body simulations**: Modeling particle interactions with equivariance constraints
- **Molecular property prediction**: Quantum chemical property estimation
- **3D point cloud analysis**: Equivariant point cloud processing
- **Physical system modeling**: Hamiltonian neural networks

### Performance Validation

Comprehensive benchmarks demonstrate improvements over:
- **cuEquivariance**: 3-5× faster for equivalent operations
- **CEGNN baseline**: 2-4× speedup with better memory efficiency
- **Standard MLPs**: Comparable performance with built-in equivariance
