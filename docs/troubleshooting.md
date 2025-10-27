# Troubleshooting Guide

This guide covers common issues, debugging techniques, and solutions for problems that may arise when using or developing Flash Clifford.

## Common Issues

### Installation Problems

#### CUDA Compatibility Issues

**Problem**: Import errors or CUDA runtime errors

**Symptoms**:
```
ImportError: libcuda.so.1: cannot open shared object file
RuntimeError: CUDA error: no kernel image is available for execution
```

**Solutions**:

1. **Check CUDA installation**:
```bash
nvidia-smi
nvcc --version
```

2. **Verify PyTorch CUDA support**:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.version.cuda)         # Should match system CUDA
```

3. **Check compute capability**:
```python
if torch.cuda.is_available():
    print(torch.cuda.get_device_capability())
    # Minimum requirement: (7, 0)
```

#### Triton Installation Issues

**Problem**: Triton import failures or compilation errors

**Solutions**:

1. **Reinstall Triton**:
```bash
pip uninstall triton -y
pip install triton>=3.0
```

2. **Check Triton version compatibility**:
```python
import triton
print(triton.__version__)  # Should be >= 3.0
```

3. **Environment variables**:
```bash
export TRITON_CACHE_DIR=/tmp/triton_cache
export TRITON_DEBUG=1  # Enable debug logging
```

### Runtime Errors

#### Memory Issues

**Problem**: CUDA out of memory errors

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate ...
```

**Debugging steps**:

1. **Check GPU memory**:
```python
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Used memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
print(f"Reserved memory: {torch.cuda.memory_reserved() / 1e9:.1f} GB")
```

2. **Monitor memory usage**:
```python
@contextmanager
def memory_monitor():
    torch.cuda.reset_peak_memory_stats()
    initial = torch.cuda.memory_allocated()
    yield
    peak = torch.cuda.max_memory_allocated()
    print(f"Memory increase: {(peak - initial) / 1e9:.3f} GB")
```

3. **Reduce batch size or feature dimensions**:
```python
# Instead of: x = torch.randn(4, 8192, 1024)
x = torch.randn(4, 2048, 512)  # Smaller size
```

#### Numerical Issues

**Problem**: NaN or Inf values in outputs

**Symptoms**:
```
tensor(nan) or tensor(inf) in output
Gradient computation fails
```

**Debugging steps**:

1. **Check input validity**:
```python
def validate_inputs(*tensors, names=None):
    """Validate tensor inputs for numerical issues."""
    if names is None:
        names = [f"tensor_{i}" for i in range(len(tensors))]

    for tensor, name in zip(tensors, names):
        if torch.isnan(tensor).any():
            raise ValueError(f"{name} contains NaN")
        if torch.isinf(tensor).any():
            raise ValueError(f"{name} contains Inf")
        if tensor.abs().max() > 1e10:
            warnings.warn(f"{name} has very large values")
```

2. **Enable anomaly detection**:
```python
with torch.autograd.detect_anomaly():
    output = model(x)  # Will stop at first NaN/Inf
```

3. **Gradient clipping**:
```python
# Add gradient clipping to prevent explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### Shape Mismatch Errors

**Problem**: Runtime shape errors

**Symptoms**:
```
RuntimeError: The size of tensor a (...) must match the size of tensor b (...)
```

**Debugging steps**:

1. **Check multivector dimensions**:
```python
# 2D Clifford: shape should be (4, batch, features)
# 3D Clifford: shape should be (8, batch, features)
print(f"Input shape: {x.shape}")
print(f"Expected: ({4 if dims==2 else 8}, batch, features)")
```

2. **Verify weight dimensions**:
```python
# SGP weights: (features, num_product_weights)
# 2D: num_product_weights = 10
# 3D: num_product_weights = 20

print(f"Weight shape: {weight.shape}")
print(f"Expected: ({features}, {10 if dims==2 else 20})")
```

3. **Check memory layout**:
```python
if not x.is_contiguous():
    x = x.contiguous()
    print("Input was not contiguous, fixed")
```

### Performance Issues

#### Slow Performance

**Problem**: Slower than expected performance

**Debugging steps**:

1. **Check GPU utilization**:
```bash
# Use nvidia-smi to monitor GPU usage during execution
watch -n 1 nvidia-smi
```

2. **Profile kernel execution**:
```python
with torch.profiler.profile() as prof:
    output = model(x)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

3. **Verify tensor placement**:
```python
print(f"Input device: {x.device}")
print(f"Model device: {next(model.parameters()).device}")
# Should both be cuda:0 or same device
```

#### Memory Bandwidth Bottlenecks

**Problem**: Performance limited by memory bandwidth

**Debugging steps**:

1. **Check memory access patterns**:
```python
# Ensure inputs are contiguous
assert x.is_contiguous(), "Input not contiguous"

# Check for efficient memory layout
print(f"Input strides: {x.stride()}")  # Should be (batch*features, features, 1)
```

2. **Monitor memory bandwidth**:
```bash
# Use nvidia-smi for bandwidth monitoring
nvidia-smi --query-gpu=utilization.memory --format=csv -l 1
```

#### Kernel Launch Overhead

**Problem**: Performance issues with small inputs

**Debugging steps**:

1. **Increase problem size**:
```python
# Small batches may have kernel launch overhead
# Try larger batch sizes: 1024, 2048, 4096
batch_size = 4096  # Instead of 64
```

2. **Batch multiple operations**:
```python
# Instead of multiple small operations
for small_batch in batches:
    result = model(small_batch)

# Use single large operation
large_batch = torch.cat(batches, dim=1)
result = model(large_batch)
```

## Development Debugging

### Triton Kernel Debugging

#### Compilation Issues

**Problem**: Triton kernel compilation fails

**Debugging steps**:

1. **Check kernel syntax**:
```python
# Use Triton's debugging features
import triton

# Enable debug mode
triton.debug = True

# Check for syntax errors
try:
    kernel_fn = triton.jit(your_kernel_function)
except Exception as e:
    print(f"Compilation error: {e}")
```

2. **Verify type annotations**:
```python
# All kernel parameters need proper type hints
@triton.jit
def kernel_function(
    x_ptr,           # tl.pointer_type
    y_ptr: tl.pointer_type,
    output_ptr: tl.pointer_type,
    n_elements: tl.int32,  # Use tl.int32, not int
    BLOCK_SIZE: tl.constexpr,
):
```

#### Runtime Kernel Issues

**Problem**: Kernels fail during execution

**Debugging steps**:

1. **Check input validation**:
```python
def debug_kernel_inputs(x, y, weight):
    """Validate kernel inputs before execution."""
    assert x.is_cuda, "x must be CUDA tensor"
    assert y.is_cuda, "y must be CUDA tensor"
    assert weight.is_cuda, "weight must be CUDA tensor"

    assert x.is_contiguous(), "x must be contiguous"
    assert y.is_contiguous(), "y must be contiguous"
    assert weight.is_contiguous(), "weight must be contiguous"

    assert x.shape == y.shape, f"Shape mismatch: {x.shape} vs {y.shape}"
```

2. **Add bounds checking**:
```python
# In kernel code
batch_ids = batch_block_id * BATCH_BLOCK + tl.arange(0, BATCH_BLOCK)
feature_ids = thread_block_id * FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)

# Add bounds checking
batch_mask = batch_ids < batch_size
feature_mask = feature_ids < n_features
```

### Gradient Debugging

#### Gradient Vanishing/Explosion

**Problem**: Gradients become very small or very large

**Debugging steps**:

1. **Monitor gradient norms**:
```python
def log_gradient_norms(model, step):
    """Log gradient norms during training."""
    total_norm = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)

    print(f"Step {step}: gradient norm = {total_norm:.6f}")
    return total_norm
```

2. **Check initialization**:
```python
# Verify parameter initialization
for name, param in model.named_parameters():
    print(f"{name}: mean={param.mean():.6f}, std={param.std():.6f}")
```

#### Gradient Checking

**Problem**: Gradients don't match finite differences

**Debugging steps**:

1. **Finite difference verification**:
```python
def finite_difference_check(model, x, epsilon=1e-6):
    """Verify gradients using finite differences."""

    # Forward pass
    output = model(x)
    loss = output.sum()

    # Compute gradients
    loss.backward()
    analytical_grads = [p.grad.clone() for p in model.parameters()]

    # Clear gradients
    for p in model.parameters():
        p.grad = None

    # Finite difference
    numerical_grads = []
    for param in model.parameters():
        param_flat = param.view(-1)
        grad_flat = torch.zeros_like(param_flat)

        for i in range(len(param_flat)):
            # Forward difference
            param_flat[i] += epsilon
            loss_plus = model(x).sum()

            param_flat[i] -= 2 * epsilon
            loss_minus = model(x).sum()

            param_flat[i] += epsilon  # Restore

            grad_flat[i] = (loss_plus - loss_minus) / (2 * epsilon)

        numerical_grads.append(grad_flat.view_as(param))

    # Compare
    for analytical, numerical in zip(analytical_grads, numerical_grads):
        diff = (analytical - numerical).abs().max()
        print(f"Max gradient difference: {diff:.6f}")
```

### Performance Debugging

#### Profiling Tools

**Problem**: Need to identify performance bottlenecks

**Tools and techniques**:

1. **PyTorch Profiler**:
```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True
) as prof:
    output = model(x)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

2. **NVIDIA Nsight Systems**:
```bash
# Profile entire application
nsys profile python train.py

# Profile specific kernels
nsys profile --output=profile_output python benchmark.py
```

3. **Custom performance monitoring**:
```python
class PerformanceMonitor:
    """Monitor performance metrics during execution."""

    def __init__(self):
        self.start_events = {}
        self.end_events = {}

    def start_timing(self, name):
        """Start timing a code section."""
        torch.cuda.synchronize()
        self.start_events[name] = torch.cuda.Event(enable_timing=True)
        self.end_events[name] = torch.cuda.Event(enable_timing=True)
        self.start_events[name].record()

    def end_timing(self, name):
        """End timing and record duration."""
        self.end_events[name].record()
        torch.cuda.synchronize()
        return self.start_events[name].elapsed_time(self.end_events[name])

    def memory_usage(self, name):
        """Record memory usage."""
        return {
            'allocated': torch.cuda.memory_allocated() / 1e6,
            'reserved': torch.cuda.memory_reserved() / 1e6,
            'peak': torch.cuda.max_memory_allocated() / 1e6
        }
```

#### Memory Leak Detection

**Problem**: GPU memory usage grows over time

**Detection techniques**:

1. **Monitor memory over iterations**:
```python
def detect_memory_leaks(model, data_loader, max_iterations=1000):
    """Detect memory leaks during training."""

    baseline_memory = torch.cuda.memory_allocated()

    for i, batch in enumerate(data_loader):
        if i >= max_iterations:
            break

        # Training step
        loss = train_step(model, batch)

        # Check memory growth
        current_memory = torch.cuda.memory_allocated()
        if current_memory > baseline_memory * 1.1:  # 10% increase
            print(f"Memory leak detected at iteration {i}")
            print(f"Memory: {baseline_memory:.1f} MB -> {current_memory:.1f} MB")
            break

        # Clear cache periodically
        if i % 100 == 0:
            torch.cuda.empty_cache()

    print(f"Final memory: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
```

2. **Check for tensor accumulation**:
```python
# Ensure gradients are cleared
optimizer.zero_grad()

# Check for unintended tensor references
import gc
if hasattr(torch.cuda, 'empty_cache'):
    torch.cuda.empty_cache()
gc.collect()
```

## Hardware-specific Issues

### Multi-GPU Issues

**Problem**: Performance issues with multiple GPUs

**Debugging steps**:

1. **Check GPU utilization**:
```python
# Use DataParallel
model = torch.nn.DataParallel(model)
output = model(x)  # Should use all GPUs

# Check individual GPU utilization
for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        print(f"GPU {i}: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
```

2. **Verify data distribution**:
```python
# Ensure batch is properly distributed
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
# Should be same shape as input
```

### Mixed Precision Issues

**Problem**: Numerical issues with mixed precision training

**Debugging steps**:

1. **Check gradient scaling**:
```python
from torch.cuda.amp import GradScaler

scaler = GradScaler()

# Monitor scale factor
print(f"Scale factor: {scaler.get_scale()}")

# Check for gradient underflow
if scaler.get_scale() > 1000:
    print("Warning: Large scale factor, possible underflow")
```

2. **Validate mixed precision accuracy**:
```python
# Compare FP32 vs FP16 results
with torch.cuda.amp.autocast():
    output_fp16 = model(x.half())

output_fp32 = model(x.float())

diff = (output_fp32 - output_fp16.float()).abs().max()
print(f"Max difference (FP32 vs FP16): {diff:.6f}")
```

## Testing and Validation

### Automated Testing

**Problem**: Tests fail or give inconsistent results

**Debugging steps**:

1. **Run tests with verbose output**:
```bash
python -m pytest tests/ -v -s --tb=short
```

2. **Check test environment**:
```python
def test_environment():
    """Verify test environment is correct."""
    assert torch.cuda.is_available(), "CUDA required"
    assert triton.__version__ >= (3, 0, 0), "Triton >= 3.0 required"

    # Check compute capability
    capability = torch.cuda.get_device_capability()
    assert capability >= (7, 0), f"Compute capability {capability} < (7, 0)"
```

3. **Reproduce failing tests**:
```python
# Run specific failing test
python -m pytest tests/test_specific.py::test_failing -v -s

# Debug with print statements
def test_failing():
    print("Starting test...")
    x = torch.randn(4, 1024, 256).cuda()
    print(f"Input: {x.shape}, {x.device}")

    result = model(x)
    print(f"Output: {result.shape}")

    # Add assertions with detailed error messages
    assert result.shape == x.shape, f"Shape mismatch: {result.shape} vs {x.shape}"
```

### Benchmarking Issues

**Problem**: Benchmarks give inconsistent or unexpected results

**Debugging steps**:

1. **Warm up properly**:
```python
# Warm up GPU and JIT compilation
for _ in range(10):
    _ = model(x)

# Synchronize before timing
torch.cuda.synchronize()
```

2. **Check for JIT compilation effects**:
```python
# First run may include compilation time
start_time = time.time()
result1 = model(x)  # Includes compilation
torch.cuda.synchronize()
compile_time = time.time() - start_time

# Subsequent runs are faster
start_time = time.time()
result2 = model(x)  # Pure execution time
torch.cuda.synchronize()
execution_time = time.time() - start_time
```

3. **Verify benchmark setup**:
```python
def validate_benchmark_setup():
    """Ensure benchmark environment is consistent."""
    assert x.is_contiguous(), "Input must be contiguous"
    assert x.device.type == 'cuda', "Must use CUDA"
    assert torch.backends.cudnn.benchmark, "Enable CUDNN benchmark"
```

## Getting Help

### Reporting Issues

When reporting issues, please include:

1. **Environment information**:
```python
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Compute capability: {torch.cuda.get_device_capability()}")
```

2. **Minimal reproduction**:
```python
# Provide minimal code that reproduces the issue
x = torch.randn(4, 1024, 256).cuda()
model = Layer(256, dims=2)
output = model(x)  # This fails with...
```

3. **Error traceback**:
```
# Full error message including stack trace
```

4. **Expected vs actual behavior**:
```
# What you expected to happen
# What actually happened
```

### Community Support

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For general questions and discussions
- **Pull Requests**: For contributions and fixes

This troubleshooting guide should help resolve most common issues. For complex problems, please provide detailed reproduction steps and environment information.

