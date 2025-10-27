# API Reference

This section provides comprehensive documentation for all public APIs in Flash Clifford, including modules, classes, and functions.

## Core Modules

### modules.layer

The main neural network layer implementation.

#### Layer

```python
class Layer(torch.nn.Module):
    """Linear layer: grade-wise linear + weighted GP + GELU + LayerNorm.

    Efficient implementation of Clifford algebra neural network layers with
    fused operations for optimal performance.

    Args:
        n_features (int): Number of features in each multivector component.
        dims (int): Dimensionality of the space (2 or 3).
        normalize (bool, optional): Whether to apply RMS normalization. Default: True.
        use_fc (bool, optional): Whether to use fully connected GP weights. Default: False.

    Attributes:
        linear_weight (torch.nn.Parameter): Linear transformation weights.
        linear_bias (torch.nn.Parameter): Linear transformation bias.
        gp_weight (torch.nn.Parameter): Geometric product weights.
        normalize (bool): Normalization flag.
        fused_op (callable): Fused operation function.
    """

    def __init__(self, n_features: int, dims: int, normalize: bool = True, use_fc: bool = False):
        """Initialize the Clifford algebra layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the layer.

        Args:
            x (torch.Tensor): Input multivector of shape (MV_DIM, batch_size, n_features).

        Returns:
            torch.Tensor: Output multivector of same shape as input.
        """
```

**Examples:**

```python
# 2D Clifford layer
layer_2d = Layer(n_features=512, dims=2, normalize=True, use_fc=False)

# 3D Clifford layer
layer_3d = Layer(n_features=256, dims=3, normalize=True, use_fc=True)

# Forward pass
x = torch.randn(4, 4096, 512)  # 2D multivector
output = layer_2d(x)

y = torch.randn(8, 2048, 256)  # 3D multivector
output = layer_3d(y)
```

### modules.baseline

Baseline implementations for comparison and testing.

#### Layer (Baseline)

```python
class Layer(torch.nn.Module):
    """Baseline linear layer implementation using PyTorch operations.

    This implementation serves as a reference for correctness verification
    and performance comparison with the optimized Triton versions.

    Args:
        n_features (int): Number of features in each multivector component.
        dims (int): Dimensionality of the space (2 or 3).
        normalize (bool, optional): Whether to apply RMS normalization. Default: True.
        use_fc (bool, optional): Whether to use fully connected GP weights. Default: False.
    """

    def __init__(self, n_features: int, dims: int, normalize: bool = True, use_fc: bool = False):
        """Initialize the baseline layer."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the baseline layer."""
```

## Operations

### ops (Operations Module)

#### 2D Operations

##### fused_gelu_sgp_norm_2d

```python
def fused_gelu_sgp_norm_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """Fused GELU + weighted geometric product + normalization for Cl(2,0).

    Applies GELU activation to inputs, computes their weighted geometric product,
    and optionally applies grade-wise RMS normalization.

    Args:
        x (torch.Tensor): First input multivector (4, batch_size, n_features).
        y (torch.Tensor): Second input multivector (4, batch_size, n_features).
        weight (torch.Tensor): GP weights (n_features, 10).
        normalize (bool): Whether to apply RMS normalization.

    Returns:
        torch.Tensor: Output multivector (4, batch_size, n_features).
    """
```

##### fused_gelu_fcgp_norm_2d

```python
def fused_gelu_fcgp_norm_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """Fused GELU + fully connected geometric product + normalization for Cl(2,0).

    Args:
        x (torch.Tensor): First input multivector (4, batch_size, n_features).
        y (torch.Tensor): Second input multivector (4, batch_size, n_features).
        weight (torch.Tensor): GP weights (10, n_features, n_features).
        normalize (bool): Whether to apply RMS normalization.

    Returns:
        torch.Tensor: Output multivector (4, batch_size, n_features).
    """
```

#### 3D Operations

##### fused_gelu_sgp_norm_3d

```python
def fused_gelu_sgp_norm_3d(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """Fused GELU + weighted geometric product + normalization for Cl(3,0).

    Args:
        x (torch.Tensor): First input multivector (8, batch_size, n_features).
        y (torch.Tensor): Second input multivector (8, batch_size, n_features).
        weight (torch.Tensor): GP weights (n_features, 20).
        normalize (bool): Whether to apply RMS normalization.

    Returns:
        torch.Tensor: Output multivector (8, batch_size, n_features).
    """
```

##### fused_gelu_fcgp_norm_3d

```python
def fused_gelu_fcgp_norm_3d(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """Fused GELU + fully connected geometric product + normalization for Cl(3,0).

    Args:
        x (torch.Tensor): First input multivector (8, batch_size, n_features).
        y (torch.Tensor): Second input multivector (8, batch_size, n_features).
        weight (torch.Tensor): GP weights (20, n_features, n_features).
        normalize (bool): Whether to apply RMS normalization.

    Returns:
        torch.Tensor: Output multivector (8, batch_size, n_features).
    """
```

## Baseline Operations

### tests.baselines

Baseline implementations for testing and comparison.

#### Activation Functions

##### mv_gelu

```python
def mv_gelu(x: torch.Tensor) -> torch.Tensor:
    """Apply GELU activation gated by scalar component.

    Args:
        x (torch.Tensor): Input multivector.

    Returns:
        torch.Tensor: Activated multivector.
    """
```

#### Normalization Functions

##### mv_rmsnorm_2d

```python
def mv_rmsnorm_2d(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization for Cl(2,0) multivectors.

    Args:
        x (torch.Tensor): Input multivector (4, ...).
        eps (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Normalized multivector.
    """
```

##### mv_rmsnorm_3d

```python
def mv_rmsnorm_3d(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization for Cl(3,0) multivectors.

    Args:
        x (torch.Tensor): Input multivector (8, ...).
        eps (float): Small constant for numerical stability.

    Returns:
        torch.Tensor: Normalized multivector.
    """
```

#### Geometric Product Functions

##### sgp_2d

```python
def sgp_2d(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted geometric product in Cl(2,0).

    Args:
        x (torch.Tensor): First multivector (4, ...).
        y (torch.Tensor): Second multivector (4, ...).
        weight (torch.Tensor): Product weights (..., 10).

    Returns:
        torch.Tensor: Product multivector (4, ...).
    """
```

##### sgp_3d

```python
def sgp_3d(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weighted geometric product in Cl(3,0).

    Args:
        x (torch.Tensor): First multivector (8, ...).
        y (torch.Tensor): Second multivector (8, ...).
        weight (torch.Tensor): Product weights (..., 20).

    Returns:
        torch.Tensor: Product multivector (8, ...).
    """
```

##### fcgp_2d

```python
def fcgp_2d(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Fully connected geometric product in Cl(2,0).

    Args:
        x (torch.Tensor): First multivector (4, ...).
        y (torch.Tensor): Second multivector (4, ...).
        weight (torch.Tensor): Transformation weights (10, ..., ...).

    Returns:
        torch.Tensor: Product multivector (4, ...).
    """
```

##### fcgp_3d

```python
def fcgp_3d(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Fully connected geometric product in Cl(3,0).

    Args:
        x (torch.Tensor): First multivector (8, ...).
        y (torch.Tensor): Second multivector (8, ...).
        weight (torch.Tensor): Transformation weights (20, ..., ...).

    Returns:
        torch.Tensor: Product multivector (8, ...).
    """
```

## Compiled Operations

### Torch-compiled Baselines

##### gelu_sgp_norm_2d_torch

```python
def gelu_sgp_norm_2d_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """Compiled baseline for 2D weighted geometric product with GELU and normalization.

    Args:
        x (torch.Tensor): First multivector (4, batch_size, n_features).
        y (torch.Tensor): Second multivector (4, batch_size, n_features).
        weight (torch.Tensor): Product weights (n_features, 10).
        normalize (bool): Whether to apply normalization.

    Returns:
        torch.Tensor: Output multivector.
    """
```

##### gelu_sgp_norm_3d_torch

```python
def gelu_sgp_norm_3d_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """Compiled baseline for 3D weighted geometric product with GELU and normalization."""
```

##### gelu_fcgp_norm_2d_torch

```python
def gelu_fcgp_norm_2d_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """Compiled baseline for 2D fully connected geometric product with GELU and normalization."""
```

##### gelu_fcgp_norm_3d_torch

```python
def gelu_fcgp_norm_3d_torch(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """Compiled baseline for 3D fully connected geometric product with GELU and normalization."""
```

## Constants and Configuration

### Multivector Dimensions

```python
MV_DIM_2D = 4  # [scalar, vector_x, vector_y, pseudoscalar]
MV_DIM_3D = 8  # [scalar, vector_x, vector_y, vector_z, bivector_xy, bivector_xz, bivector_yz, pseudoscalar]
```

### Number of Product Weights

```python
P2M0_NUM_PRODUCT_WEIGHTS = 10  # 2D geometric product components
P3M0_NUM_PRODUCT_WEIGHTS = 20  # 3D geometric product components
```

### Number of Grades

```python
P2M0_NUM_GRADES = 3  # scalar, vector, pseudoscalar
P3M0_NUM_GRADES = 4  # scalar, vector, bivector, pseudoscalar
```

### Weight Expansion Maps

```python
# 2D: maps component index to grade index
P2M0_WEIGHT_EXPANSION = [0, 1, 1, 2]  # scalar→0, vector→1, pseudoscalar→2

# 3D: maps component index to grade index
P3M0_WEIGHT_EXPANSION = [0, 1, 1, 1, 2, 2, 2, 3]  # scalar→0, vector→1, bivector→2, pseudoscalar→3
```

## Error Handling

All functions include comprehensive error checking:

- **Shape validation**: Ensures tensors have correct dimensions
- **Device consistency**: Checks that all tensors are on the same device
- **Memory layout**: Verifies contiguous memory layout for efficiency
- **Dimensionality**: Validates that dimensions match expected Clifford algebra

## Performance Notes

### Memory Layout Requirements

All multivector tensors must use the `(MV_DIM, BATCH_SIZE, NUM_FEATURES)` layout for optimal performance.

### Device Support

Operations are optimized for CUDA devices and require:
- PyTorch with CUDA support
- Triton >= 3.0
- Compute capability 7.0 or higher

### Gradient Support

All operations support automatic differentiation through custom backward implementations that match the mathematical derivatives of the Clifford algebra operations.

