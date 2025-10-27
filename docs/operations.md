# Operations API

This section documents the core operations implemented in Flash Clifford, focusing on the fused kernels that combine multiple operations for efficiency.

## Fused Operations

Flash Clifford implements several fused operations that combine activation functions, geometric products, and normalization in single GPU kernels. This approach eliminates intermediate memory allocations and reduces kernel launch overhead.

### fused_gelu_sgp_norm_nd

Applies GELU activation → weighted geometric product → optional RMS normalization in a single fused kernel.

**2D and 3D versions:**

```python
def fused_gelu_sgp_norm_2d(x, y, weight, normalize=True):
```

```python
def fused_gelu_sgp_norm_3d(x, y, weight, normalize=True):
```

**Parameters:**
- `x` (torch.Tensor): Input multivector of shape `(MV_DIM, BATCH_SIZE, NUM_FEATURES)`
- `y` (torch.Tensor): Input multivector of shape `(MV_DIM, BATCH_SIZE, NUM_FEATURES)`
- `weight` (torch.Tensor): Geometric product weights of shape `(NUM_FEATURES, NUM_PRODUCT_WEIGHTS)`
- `normalize` (bool): Whether to apply grade-wise RMS normalization

**Returns:**
- torch.Tensor: Output multivector of shape `(MV_DIM, BATCH_SIZE, NUM_FEATURES)`

**Implementation Details:**
- **Memory layout**: Uses `(MV_DIM, BATCH, FEATURES)` layout for optimal coalescing
- **Block configuration**: `BATCH_BLOCK=4, FEATURE_BLOCK=128, NUM_WARPS=16`
- **Atomic operations**: Grade-wise normalization statistics via atomic adds
- **Gradient computation**: Custom backward pass with proper chain rule

### fused_gelu_fcgp_norm_nd

Applies GELU activation → fully connected geometric product → optional RMS normalization.

**2D and 3D versions:**

```python
def fused_gelu_fcgp_norm_2d(x, y, weight, normalize=True):
```

```python
def fused_gelu_fcgp_norm_3d(x, y, weight, normalize=True):
```

**Parameters:**
- `x` (torch.Tensor): Input multivector of shape `(MV_DIM, BATCH_SIZE, NUM_FEATURES)`
- `y` (torch.Tensor): Input multivector of shape `(MV_DIM, BATCH_SIZE, NUM_FEATURES)`
- `weight` (torch.Tensor): Geometric product weights of shape `(NUM_PRODUCT_WEIGHTS, NUM_FEATURES, NUM_FEATURES)`
- `normalize` (bool): Whether to apply grade-wise RMS normalization

**Returns:**
- torch.Tensor: Output multivector of shape `(MV_DIM, BATCH_SIZE, NUM_FEATURES)`

## Geometric Product Types

### Weighted Geometric Product (SGP)

The **weighted geometric product** applies learned weights to each component of the geometric product:

```
o = Σᵢ wᵢ * (x ⧫ y)ᵢ
```

where `wᵢ` are learnable parameters and `(x ⧫ y)ᵢ` are the components of the geometric product.

**Weight dimensions:**
- 2D: `(NUM_FEATURES, 10)` - 10 unique product components
- 3D: `(NUM_FEATURES, 20)` - 20 unique product components

### Fully Connected Geometric Product (FCGP)

The **fully connected geometric product** applies a separate linear transformation to each product component:

```
o = Σᵢ Wᵢ @ (x ⧫ y)ᵢ
```

where `Wᵢ` are learned weight matrices.

**Weight dimensions:**
- 2D: `(10, NUM_FEATURES, NUM_FEATURES)` - 10 transformation matrices
- 3D: `(20, NUM_FEATURES, NUM_FEATURES)` - 20 transformation matrices

## Activation Functions

### Multivector GELU

The GELU activation is gated by the scalar component of the multivector:

```
Φ(x) = 0.5 * (1 + erf(x₀ / √2))
y = x * Φ(x₀)
```

where `x₀` is the scalar component. This preserves the multivector structure while applying non-linearity.

## Normalization

### Grade-wise RMS Normalization

RMS normalization is applied separately to each grade of the multivector:

```
x̂ᵢ = xᵢ / √(μᵢ² + ε)
```

where:
- `μᵢ²` is the mean squared value across features for grade `i`
- `ε` is a small constant for numerical stability
- Grade-wise normalization preserves equivariance properties

## Kernel Fusion Benefits

The fused operations provide several performance advantages:

1. **Reduced memory traffic**: No intermediate tensors stored in global memory
2. **Fewer kernel launches**: Single kernel vs multiple sequential operations
3. **Better cache locality**: Operations performed on registers when possible
4. **Eliminated synchronization**: No need to wait between operations

## Implementation Details

### Memory Layout

All operations use the `(MV_DIM, BATCH_SIZE, NUM_FEATURES)` layout, which enables:

- Efficient batch matrix multiplication for linear layers
- Coalesced memory access patterns in CUDA
- Vectorized operations across the feature dimension

### Triton Optimizations

The kernels are optimized using several Triton-specific techniques:

- **Block-based computation**: Configurable block sizes for different hardware
- **Load masking**: Efficient handling of boundary conditions
- **Atomic operations**: Lock-free accumulation for normalization statistics
- **Custom gradient computation**: Optimized backward pass implementation

## Performance Characteristics

### Runtime Complexity

| Operation | 2D Complexity | 3D Complexity |
|-----------|---------------|---------------|
| SGP Forward | O(BNF) | O(BNF) |
| FCGP Forward | O(BN²F) | O(BN²F) |
| Backward Pass | 3× forward | 3× forward |

where:
- `B`: batch size
- `N`: number of features
- `F`: number of product weights

### Memory Usage

| Operation | 2D Parameters | 3D Parameters |
|-----------|---------------|---------------|
| SGP | 10N | 20N |
| FCGP | 10N² | 20N² |

## Usage Examples

### Basic Geometric Product

```python
import torch
from ops import fused_gelu_sgp_norm_2d

# Input multivectors
x = torch.randn(4, 4096, 512).cuda()  # 2D multivectors
y = torch.randn(4, 4096, 512).cuda()
weight = torch.randn(512, 10).cuda()  # 10 weights for 2D GP

# Apply fused operation
output = fused_gelu_sgp_norm_2d(x, y, weight, normalize=True)
```

### Fully Connected Variant

```python
from ops import fused_gelu_fcgp_norm_3d

# 3D multivectors with fully connected weights
x = torch.randn(8, 4096, 512).cuda()  # 3D multivectors
y = torch.randn(8, 4096, 512).cuda()
weight = torch.randn(20, 512, 512).cuda()  # 20 transformation matrices

output = fused_gelu_fcgp_norm_3d(x, y, weight, normalize=True)
```

## Gradient Computation

All operations support automatic differentiation through custom backward implementations:

- **Forward pass**: Computes the fused operation efficiently
- **Backward pass**: Computes gradients w.r.t. inputs and weights
- **Memory efficient**: Reuses intermediate values from forward pass

The gradient computation handles the complex chain rule through the fused operations, ensuring numerical stability and correctness.

### Backward Pass Complexity

The backward pass requires computing gradients through:

1. **RMS normalization**: `∂L/∂x = ∂L/∂x̂ - correction_term`
2. **Geometric product**: `∂L/∂wᵢⱼ = Σₖ ∂L/∂oₖ · ∂oₖ/∂wᵢⱼ`
3. **GELU activation**: `∂L/∂x₀ = ∂L/∂gate · ∂gate/∂x₀`

### Numerical Stability

Gradient computation maintains stability through:

- **Mixed precision accumulation**: FP32 accumulation for weight gradients
- **Atomic synchronization**: Proper synchronization of gradient accumulation
- **Numerical checks**: Detection of NaN/inf values in debug builds

### Memory Management

The backward implementation optimizes memory usage:

- **In-place operations**: Reuses forward pass buffers where possible
- **Gradient accumulation**: Reduces memory by accumulating gradients incrementally
- **Context management**: Efficient saving of intermediate values for backward pass

## Kernel Internals

### Triton JIT Compilation

All kernels use Triton's just-in-time compilation for optimal performance:

```python
@triton.jit
def gelu_wgp_norm_kernel_fwd(
    x_ptr, y_ptr, output_ptr, weights_ptr, pnorm_ptr,
    NORMALIZE: tl.constexpr, batch_size: tl.constexpr, n_features: tl.constexpr,
    BATCH_BLOCK: tl.constexpr, FEATURE_BLOCK: tl.constexpr,
    NUM_PRODUCT_WEIGHTS: tl.constexpr,
):
```

**Compile-time constants** enable:
- Loop unrolling and optimization
- Memory layout specialization
- Hardware-specific code generation

### Memory Access Patterns

The kernels implement optimal memory access patterns:

```python
# Coalesced access in feature dimension
stride_component = batch_size * n_features
base_offset = batch_ids[:, None] * n_features + feature_ids[None, :]

# Load with masking for boundary conditions
x0 = tl.load(x_ptr + 0 * stride_component + base_offset, mask=batch_feature_mask)
```

**Key optimizations:**
- **Coalescing**: Contiguous access in feature dimension
- **Broadcasting**: Efficient weight reuse across batch elements
- **Masking**: Boundary handling without branching

### Block-based Parallelism

Computation is organized in hierarchical blocks:

```python
# Grid configuration
batch_blocks = triton.cdiv(batch_size, BATCH_BLOCK)
feature_blocks = triton.cdiv(n_features, FEATURE_BLOCK)
grid = (batch_blocks, feature_blocks)

# Each block processes a tile
batch_ids = batch_block_id * BATCH_BLOCK + tl.arange(0, BATCH_BLOCK)
feature_ids = thread_block_id * FEATURE_BLOCK + tl.arange(0, FEATURE_BLOCK)
```

This enables:
- **Load balancing**: Even distribution across GPU cores
- **Shared memory usage**: Weight broadcasting within blocks
- **Scalability**: Performance scales with GPU resources

### Register-level Optimization

The kernels minimize register pressure through:

- **Variable reuse**: Computing multiple expressions with same intermediates
- **Type specialization**: Using appropriate precision for different operations
- **Loop unrolling**: Expanding small loops for better performance

### Synchronization Points

The implementation carefully manages synchronization:

- **Atomic operations**: Lock-free accumulation for normalization statistics
- **Memory barriers**: Ensuring visibility of writes between kernels
- **Stream synchronization**: Proper ordering in multi-stream scenarios
