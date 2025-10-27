# Development Guide

This guide provides comprehensive information for developers working with Flash Clifford, including debugging tools, testing frameworks, contribution guidelines, and development workflows.

## Getting Started

### Development Environment Setup

#### Prerequisites

```bash
# Core dependencies
pip install torch triton

# Development dependencies
pip install pytest black isort flake8 mypy numpy matplotlib

# Optional: CUDA development tools
pip install cuda-python nvtx
```

#### Project Structure

```
flash-clifford/
├── modules/           # Neural network layer implementations
│   ├── layer.py      # Optimized Layer class
│   └── baseline.py   # Reference implementations
├── ops/              # Triton kernel operations
│   ├── p2m0.py       # 2D geometric product kernels
│   ├── p3m0.py       # 3D geometric product kernels
│   ├── fc_p2m0.py    # 2D fully connected kernels
│   └── fc_p3m0.py    # 3D fully connected kernels
├── tests/            # Comprehensive test suite
│   ├── benchmarks/   # Performance benchmarks
│   ├── utils.py      # Testing utilities
│   ├── baselines.py  # Baseline implementations
│   └── correctness/  # Correctness verification
├── docs/             # Documentation (this directory)
├── setup.py         # Package configuration
└── pyproject.toml   # Development configuration
```

### Code Style and Quality

#### Formatting

```bash
# Format Python code
black modules/ ops/ tests/
isort modules/ ops/ tests/

# Format documentation
mdformat docs/

# Check formatting
black --check modules/ ops/ tests/
isort --check-only modules/ ops/ tests/
```

#### Linting

```bash
# Run linters
flake8 modules/ ops/ tests/
mypy modules/ ops/

# Fix common issues
autoflake --remove-all-unused-imports --remove-unused-variables -r modules/ ops/ tests/
```

#### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

## Testing Framework

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_correctness.py -v  # Correctness tests
python -m pytest tests/test_benchmarks.py -v    # Benchmark tests

# Run with coverage
python -m pytest tests/ --cov=modules --cov-report=html
```

### Test Categories

#### 1. Correctness Tests

Verify numerical accuracy against baseline implementations:

```python
def test_forward_correctness():
    """Test forward pass accuracy."""
    x = torch.randn(4, 1024, 256).cuda()
    y = torch.randn(4, 1024, 256).cuda()
    weight = torch.randn(256, 10).cuda()

    output_triton = fused_gelu_sgp_norm_2d(x, y, weight)
    output_baseline = gelu_sgp_norm_2d_torch(x, y, weight)

    assert torch.allclose(output_triton, output_baseline, atol=1e-5)

def test_backward_correctness():
    """Test backward pass gradients."""
    x = torch.randn(4, 1024, 256).cuda().requires_grad_(True)
    y = torch.randn(4, 1024, 256).cuda().requires_grad_(True)
    weight = torch.randn(256, 10).cuda().requires_grad_(True)

    # Compute gradients
    output = fused_gelu_sgp_norm_2d(x, y, weight)
    loss = output.sum()
    loss.backward()

    # Verify gradients are finite and reasonable
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(y.grad).all()
    assert torch.isfinite(weight.grad).all()
```

#### 2. Performance Tests

Benchmark against baseline implementations:

```python
def test_performance_regression():
    """Ensure performance doesn't regress."""
    setup = lambda b, n: (torch.randn(4, b, n).cuda(),
                         torch.randn(4, b, n).cuda(),
                         torch.randn(n, 10).cuda())

    result = run_single_benchmark(
        fused_gelu_sgp_norm_2d,
        gelu_sgp_norm_2d_torch,
        setup, 4096, 512, rep=1000
    )

    # Performance should be at least 2x baseline
    assert result['speedup_fwd'] >= 2.0
    assert result['speedup_fwd_bwd'] >= 2.0
```

#### 3. Integration Tests

Test end-to-end functionality:

```python
def test_layer_integration():
    """Test Layer class integration."""
    layer = Layer(n_features=256, dims=3, normalize=True)

    # Test different batch sizes and configurations
    for batch_size in [1, 1024, 4096]:
        x = torch.randn(8, batch_size, 256).cuda()
        output = layer(x)

        assert output.shape == x.shape
        assert torch.isfinite(output).all()
```

### Custom Test Utilities

#### Debugging Tools

```python
def debug_kernel_execution(kernel_fn, *args, **kwargs):
    """Debug kernel execution with detailed logging."""

    print(f"Running kernel: {kernel_fn.__name__}")
    print(f"Input shapes: {[arg.shape for arg in args]}")
    print(f"Input dtypes: {[arg.dtype for arg in args]}")

    # Check memory layout
    for i, arg in enumerate(args):
        if not arg.is_contiguous():
            print(f"Warning: Input {i} is not contiguous")

    # Run with error checking
    try:
        output = kernel_fn(*args, **kwargs)
        print(f"Output shape: {output.shape}")
        return output
    except Exception as e:
        print(f"Kernel execution failed: {e}")
        raise

def compare_implementations(triton_fn, torch_fn, *args, rtol=1e-5, atol=1e-5):
    """Compare Triton and PyTorch implementations."""

    # Forward pass comparison
    out_triton = triton_fn(*args)
    out_torch = torch_fn(*args)

    max_diff = (out_torch - out_triton).abs().max()
    is_close = torch.allclose(out_torch, out_triton, rtol=rtol, atol=atol)

    print(f"Max difference: {max_diff:.2e}")
    print(f"Close within tolerance: {is_close}")

    if not is_close:
        print("Detailed comparison:")
        print(f"Triton output: {out_triton.flatten()[:10]}")
        print(f"PyTorch output: {out_torch.flatten()[:10]}")

    return is_close, max_diff
```

## Benchmarking Framework

### Automated Benchmarking

```bash
# Run comprehensive benchmarks
python -m tests.benchmarks.layer_2d --output results_2d.json
python -m tests.benchmarks.layer_3d --output results_3d.json

# Generate performance plots
python -m tests.benchmarks.plot_results results_2d.json results_3d.json
```

### Custom Benchmarking

```python
from tests.utils import run_sweep, plot_heatmap

def benchmark_custom_configuration():
    """Benchmark custom layer configuration."""

    def setup_fn(batch_size, num_features):
        x = torch.randn(4, batch_size, num_features).cuda()
        y = torch.randn(4, batch_size, num_features).cuda()
        weight = torch.randn(num_features, 10).cuda()
        return x, y, weight

    # Parameter sweep
    results = run_sweep(
        fused_gelu_sgp_norm_2d,
        gelu_sgp_norm_2d_torch,
        setup_fn,
        batch_sizes=[1024, 2048, 4096, 8192],
        num_features_list=[128, 256, 512, 1024],
        rep=1000
    )

    # Generate visualization
    plot_heatmap(results, 'speedup_fwd', '2D SGP Forward Speedup',
                'benchmarks/sgp_2d_speedup.png')
```

### Performance Regression Testing

```python
def test_performance_regression():
    """Ensure performance doesn't regress from baseline."""

    # Store baseline performance
    baseline_results = {
        'speedup_fwd': 3.2,
        'speedup_bwd': 2.8,
        'memory_reduction': 0.31
    }

    # Run current performance tests
    current_results = run_performance_tests()

    # Check for regressions
    for metric in baseline_results:
        if current_results[metric] < baseline_results[metric] * 0.9:
            raise AssertionError(f"Performance regression in {metric}")
```

## Debugging and Troubleshooting

### Common Issues

#### 1. CUDA Errors

```python
# Check CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available")

# Check compute capability
if torch.cuda.get_device_capability()[0] < 7:
    warnings.warn("Compute capability < 7.0 may have reduced performance")

# Memory issues
try:
    output = kernel_fn(*large_tensors)
except RuntimeError as e:
    if "out of memory" in str(e):
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory}")
        print(f"Used memory: {torch.cuda.memory_allocated()}")
        raise
```

#### 2. Numerical Issues

```python
def check_numerical_stability(tensors, names):
    """Check for numerical issues in tensors."""

    issues = []
    for tensor, name in zip(tensors, names):
        if torch.isnan(tensor).any():
            issues.append(f"{name}: contains NaN")
        if torch.isinf(tensor).any():
            issues.append(f"{name}: contains Inf")
        if tensor.abs().max() > 1e6:
            issues.append(f"{name}: very large values")

    if issues:
        raise ValueError(f"Numerical issues detected: {issues}")
```

#### 3. Gradient Issues

```python
def debug_gradients(model, inputs, loss_fn):
    """Debug gradient computation."""

    # Enable anomaly detection
    with torch.autograd.detect_anomaly():
        outputs = model(inputs)
        loss = loss_fn(outputs)
        loss.backward()

    # Check gradient statistics
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"No gradient for {name}")
        else:
            print(f"{name}: grad norm = {param.grad.norm():.6f}")
            if torch.isnan(param.grad).any():
                print(f"WARNING: {name} has NaN gradients")
```

### Kernel Debugging

#### Triton Kernel Debugging

```python
def debug_triton_kernel(kernel_fn, *args, verbose=True):
    """Debug Triton kernel execution."""

    if verbose:
        print(f"Debugging kernel: {kernel_fn.__name__}")
        for i, arg in enumerate(args):
            print(f"  Arg {i}: shape={arg.shape}, dtype={arg.dtype}, "
                  f"device={arg.device}, contiguous={arg.is_contiguous()}")

    # Run kernel with error handling
    try:
        result = kernel_fn(*args)
        if verbose:
            print(f"  Output: shape={result.shape}, dtype={result.dtype}")
        return result
    except Exception as e:
        print(f"  Kernel failed: {e}")
        # Print kernel source for debugging
        if hasattr(kernel_fn, 'debug_kernel'):
            print("  Kernel source:")
            print(kernel_fn.debug_kernel)
        raise
```

#### GPU Memory Debugging

```python
@contextmanager
def cuda_memory_debug():
    """Context manager for CUDA memory debugging."""

    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()

    try:
        yield
    finally:
        peak_memory = torch.cuda.max_memory_allocated()
        final_memory = torch.cuda.memory_allocated()

        print(f"Memory allocated: {initial_memory / 1e6:.1f} MB")
        print(f"Peak memory: {peak_memory / 1e6:.1f} MB")
        print(f"Final memory: {final_memory / 1e6:.1f} MB")
        print(f"Memory increase: {(final_memory - initial_memory) / 1e6:.1f} MB")
```

## Contributing Guidelines

### Code Contributions

#### 1. Feature Branches

```bash
# Create feature branch
git checkout -b feature/new-geometric-product

# Make changes
# Add tests
# Update documentation

# Submit pull request
git add .
git commit -m "Add new geometric product implementation"
git push origin feature/new-geometric-product
```

#### 2. Testing Requirements

All contributions must include:

- **Unit tests** for new functionality
- **Integration tests** for API changes
- **Performance benchmarks** for optimizations
- **Documentation updates** for new features

```python
def test_new_feature():
    """Test new feature implementation."""
    # Arrange
    inputs = create_test_inputs()

    # Act
    outputs = new_feature_function(inputs)

    # Assert
    assert outputs.shape == expected_shape
    assert torch.allclose(outputs, expected_outputs)
```

#### 3. Performance Validation

```python
def validate_performance_impact():
    """Validate that changes don't break performance."""

    # Benchmark before changes (store as baseline)
    baseline = benchmark_function()

    # Apply changes
    apply_changes()

    # Benchmark after changes
    current = benchmark_function()

    # Ensure no regressions
    assert current.time <= baseline.time * 1.1  # 10% tolerance
    assert current.memory <= baseline.memory * 1.05  # 5% tolerance
```

### Documentation Contributions

#### 1. Documentation Standards

```markdown
# Clear section headers
## Technical Details

# Code examples with proper syntax highlighting
```python
def example_function(x: torch.Tensor) -> torch.Tensor:
    """Example function with docstring."""
    return x * 2
```

# Mathematical formulations
The geometric product satisfies: `AB = A · B + A ∧ B`

# Performance notes
**Performance**: O(n) complexity with GPU acceleration
```

#### 2. API Documentation

```python
def new_function(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    normalize: bool = True
) -> torch.Tensor:
    """Complete function documentation.

    This function implements the new geometric operation with
    detailed mathematical formulation.

    Args:
        x: First input multivector of shape (MV_DIM, BATCH, FEATURES)
        y: Second input multivector with same shape as x
        weight: Optional weight tensor for weighted products
        normalize: Whether to apply normalization

    Returns:
        Output multivector with same shape as inputs

    Raises:
        ValueError: If input shapes are incompatible
        RuntimeError: If CUDA operations fail

    Example:
        >>> x = torch.randn(4, 1024, 256)
        >>> y = torch.randn(4, 1024, 256)
        >>> output = new_function(x, y)
        >>> print(output.shape)
        torch.Size([4, 1024, 256])
    """
```

### Review Process

#### 1. Code Review Checklist

- [ ] Code follows style guidelines (black, isort)
- [ ] All tests pass
- [ ] Performance benchmarks included
- [ ] Documentation updated
- [ ] Type hints included
- [ ] Error handling implemented
- [ ] Backward compatibility maintained

#### 2. Performance Review

- [ ] Benchmark results show improvement or justify regression
- [ ] Memory usage analysis provided
- [ ] Hardware compatibility verified
- [ ] Scaling analysis included

#### 3. Documentation Review

- [ ] API documentation complete
- [ ] Examples provided
- [ ] Mathematical formulations correct
- [ ] Performance notes included

## Development Workflow

### 1. Feature Development

```bash
# 1. Create branch
git checkout -b feature/new-kernel

# 2. Implement feature
# Add Triton kernel in ops/
# Add Python wrapper in modules/
# Add tests in tests/

# 3. Test thoroughly
python -m pytest tests/test_new_feature.py -v
python -m tests.benchmarks.new_feature_benchmark

# 4. Update documentation
# Add to docs/api-reference.md
# Add examples to docs/examples.md

# 5. Format and lint
black . && isort . && flake8 .

# 6. Commit and push
git add .
git commit -m "Add new kernel implementation"
git push origin feature/new-kernel
```

### 2. Bug Fixes

```bash
# 1. Reproduce issue
python -m pytest tests/ -k "test_failing" -v

# 2. Create fix branch
git checkout -b fix/bug-description

# 3. Implement fix
# Modify relevant files
# Add regression test

# 4. Verify fix
python -m pytest tests/test_regression.py -v

# 5. Update changelog
echo "- Fix: Description of fix" >> CHANGELOG.md

# 6. Commit and push
git add .
git commit -m "Fix: bug description"
git push origin fix/bug-description
```

### 3. Performance Improvements

```bash
# 1. Profile current performance
python -m tests.benchmarks.profile_current

# 2. Implement optimization
# Modify kernels or algorithms

# 3. Benchmark improvement
python -m tests.benchmarks.profile_optimized

# 4. Verify correctness maintained
python -m pytest tests/test_correctness.py -v

# 5. Update performance docs
# Add results to docs/performance.md

# 6. Commit with performance data
git add .
git commit -m "Optimize: 2.5x speedup in geometric product"
git push origin optimize/geometric-product
```

## Advanced Development Tools

### 1. Custom Triton Debugging

```python
class TritonDebugger:
    """Advanced debugging for Triton kernels."""

    def __init__(self, kernel_fn):
        self.kernel_fn = kernel_fn
        self.debug_info = {}

    def debug_execution(self, *args, **kwargs):
        """Debug kernel execution step by step."""

        # Check inputs
        self.validate_inputs(args)

        # Print kernel configuration
        self.print_kernel_config(kwargs)

        # Execute with profiling
        with torch.profiler.profile() as prof:
            result = self.kernel_fn(*args, **kwargs)

        # Analyze performance
        self.analyze_performance(prof)

        return result

    def validate_inputs(self, args):
        """Validate kernel inputs."""
        for i, arg in enumerate(args):
            assert arg.is_cuda, f"Input {i} must be CUDA tensor"
            assert arg.is_contiguous(), f"Input {i} must be contiguous"
            assert arg.dtype in [torch.float32, torch.float16], f"Unsupported dtype: {arg.dtype}"

    def analyze_performance(self, profiler):
        """Analyze kernel performance."""
        events = profiler.events()
        kernel_events = [e for e in events if 'triton' in e.name.lower()]

        for event in kernel_events:
            print(f"Kernel: {event.name}")
            print(f"  Duration: {event.duration}")
            print(f"  Memory: {event.memory}")
```

### 2. Automatic Differentiation Debugging

```python
def debug_autograd(function, inputs, output_grad=None):
    """Debug automatic differentiation."""

    # Enable gradient computation
    inputs = [x.requires_grad_(True) for x in inputs]

    # Forward pass
    outputs = function(*inputs)

    # Backward pass
    if output_grad is None:
        output_grad = torch.ones_like(outputs)

    grads = torch.autograd.grad(outputs, inputs, output_grad,
                               retain_graph=True, create_graph=True)

    # Analyze gradients
    for i, (inp, grad) in enumerate(zip(inputs, grads)):
        print(f"Input {i}:")
        print(f"  Shape: {inp.shape}")
        print(f"  Gradient norm: {grad.norm():.6f}")
        print(f"  Gradient max: {grad.max():.6f}")
        print(f"  Gradient min: {grad.min():.6f}")

    return outputs, grads
```

### 3. Memory Leak Detection

```python
def detect_memory_leaks(function, *args, iterations=100):
    """Detect memory leaks in function."""

    initial_memory = torch.cuda.memory_allocated()

    for i in range(iterations):
        # Clear cache periodically
        if i % 10 == 0:
            torch.cuda.empty_cache()

        result = function(*args)

        # Check for memory growth
        current_memory = torch.cuda.memory_allocated()
        if current_memory > initial_memory * 1.1:  # 10% growth
            print(f"Potential memory leak detected at iteration {i}")
            print(f"Memory: {initial_memory} -> {current_memory}")
            break

        # Clean up
        del result
        torch.cuda.empty_cache()

    final_memory = torch.cuda.memory_allocated()
    print(f"Memory usage: {initial_memory} -> {final_memory}")
```

This development guide provides comprehensive tools and workflows for contributing to and maintaining the Flash Clifford library. The emphasis on testing, performance validation, and documentation ensures high code quality and maintainability.

