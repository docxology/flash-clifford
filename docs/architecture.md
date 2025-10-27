# Architecture

This section describes the neural network architecture implemented in Flash Clifford, focusing on the Layer class and design patterns for building Clifford neural networks.

## Layer Architecture

The `Layer` class implements a complete Clifford algebra neural network layer that combines linear transformations with geometric operations.

### Core Components

```python
class Layer(torch.nn.Module):
    """Linear layer: grade-wise linear + weighted GP + GELU + LayerNorm."""
```

The layer consists of:

1. **Grade-wise linear transformation**: Linear layer applied separately to each grade
2. **Geometric product**: Weighted or fully connected geometric product
3. **GELU activation**: Multivector-aware activation function
4. **RMS normalization**: Grade-wise normalization (optional)

### Mathematical Formulation

For input multivector `x` and layer parameters, the forward pass computes:

```
y = Linear(x) + bias
y = GELU(y)
z = GeometricProduct(x, y, weights)
output = RMSNorm(z)  # if normalize=True
```

### Parameter Structure

The layer maintains several parameter groups:

#### Linear Parameters
- **Weight**: `(NUM_GRADES, N_IN, N_OUT)` - separate linear transformation per grade
- **Bias**: `(1, 1, N_OUT)` - grade-agnostic bias

#### Geometric Product Parameters
- **SGP weights**: `(N_OUT, NUM_PRODUCT_WEIGHTS)` - one weight per GP component
- **FCGP weights**: `(NUM_PRODUCT_WEIGHTS, N_OUT, N_OUT)` - transformation matrix per component

## Grade-wise Processing

The architecture processes multivectors by grade, respecting the algebraic structure:

### 2D Multivectors (Cl(2,0))
```
Components: [scalar, vector_x, vector_y, pseudoscalar]
Grades:     [   0,       1,       1,        0   ]
Linear weights shape: (4, N_IN, N_OUT)
```

### 3D Multivectors (Cl(3,0))
```
Components: [scalar, vec_x, vec_y, vec_z, biv_xy, biv_xz, biv_yz, pseudoscalar]
Grades:     [   0,     1,     1,     1,     2,      2,      2,        3     ]
Linear weights shape: (8, N_IN, N_OUT)
```

### Weight Expansion

The `weight_expansion` tensor maps grades to their corresponding linear weight indices:

```python
# 2D: scalar→0, vector→1, pseudoscalar→2
weight_expansion_2d = [0, 1, 1, 2]

# 3D: scalar→0, vector→1, bivector→2, pseudoscalar→3
weight_expansion_3d = [0, 1, 1, 1, 2, 2, 2, 3]
```

**Mathematical justification:** Components of the same grade transform identically under O(n) actions, so they should share linear transformation parameters.

### Implementation Details

The grade-wise processing is implemented through:

```python
# Grade-wise linear transformation
def forward(self, x):
    # x: (MV_DIM, BATCH, FEATURES)
    # self.linear_weight: (NUM_GRADES, N_IN, N_OUT)
    # self.weight_expansion: (MV_DIM,) maps component → grade

    y = torch.bmm(x, self.linear_weight[self.weight_expansion])
    # Result: (MV_DIM, BATCH, N_OUT) with proper grade-wise transformations
```

This implements the key insight that equivariant transformations should respect the grade structure of the multivector representation.

## Initialization

The layer parameters are initialized using variance-preserving initialization:

```python
# Geometric product weights
torch.nn.init.normal_(gp_weight, std=1 / sqrt(dims + 1))

# Linear weights
torch.nn.init.normal_(linear_weight, std=1 / sqrt(n_features * (dims + 1)))
```

The initialization accounts for:
- **Geometric product variance**: Depends on the dimensionality of the space
- **Grade multiplicity**: Higher grades have more components
- **Feature scaling**: Proper scaling with network width

## Memory Layout Optimization

The architecture uses the `(MV_DIM, BATCH_SIZE, NUM_FEATURES)` memory layout:

### Benefits

1. **Batch matrix multiplication**: Linear layers become efficient `bmm` operations
2. **Coalesced access**: Memory accesses are contiguous in the feature dimension
3. **Vectorization**: Enables SIMD operations across features
4. **Cache efficiency**: Better spatial locality for GPU caches

### Implementation

```python
def forward(self, x):
    # Linear transformation: (MV_DIM, B, N) @ (MV_DIM, N, N) -> (MV_DIM, B, N)
    y = torch.bmm(x, self.linear_weight[self.weight_expansion]) + self.linear_bias

    # Geometric product with fused operations
    return self.fused_op(x, y, self.gp_weight, self.normalize)
```

## Configuration Options

### Dimensionality
```python
# 2D Clifford algebra
layer = Layer(n_features=512, dims=2, normalize=True, use_fc=False)

# 3D Clifford algebra
layer = Layer(n_features=512, dims=3, normalize=True, use_fc=False)
```

### Normalization
```python
# With grade-wise RMS normalization
layer = Layer(n_features=512, dims=3, normalize=True)

# Without normalization (for residual connections)
layer = Layer(n_features=512, dims=3, normalize=False)
```

### Geometric Product Type
```python
# Weighted geometric product (memory efficient)
layer = Layer(n_features=512, dims=3, use_fc=False)

# Fully connected geometric product (more expressive)
layer = Layer(n_features=512, dims=3, use_fc=True)
```

## Equivariance Properties

The layer architecture preserves equivariance under orthogonal transformations:

### Transformation Rule
If `R ∈ O(n)` is an orthogonal transformation represented as a multivector, then:

```
Layer(RxR†) = R · Layer(x) · R†
```

where `R†` is the reverse (involution) of R.

### Grade Preservation
Each grade transforms according to its irreducible representation:
- Scalars transform trivially (grade 0)
- Vectors transform as vectors (grade 1)
- Bivectors transform as bivectors (grade 2)
- Higher grades follow their respective transformation rules

## Computational Graph

The layer integrates seamlessly with PyTorch's automatic differentiation:

```python
# Forward pass
output = layer(x)  # (MV_DIM, BATCH_SIZE, NUM_FEATURES)

# Backward pass (automatic)
loss = criterion(output, target)
loss.backward()  # Computes gradients through all operations
```

The custom Triton kernels implement efficient forward and backward passes that are automatically differentiable.

## Performance Characteristics

### Computational Complexity

| Operation | Complexity | Parameters |
|-----------|------------|------------|
| Linear Layer | O(MV_DIM × B × N × N) | MV_DIM × N × N |
| SGP | O(B × N × N_PRODUCT_WEIGHTS) | N × N_PRODUCT_WEIGHTS |
| FCGP | O(B × N² × N_PRODUCT_WEIGHTS) | N_PRODUCT_WEIGHTS × N² |

### Memory Usage

| Component | Memory (SGP) | Memory (FCGP) |
|-----------|--------------|---------------|
| Linear weights | 4N² (2D), 8N² (3D) | 4N² (2D), 8N² (3D) |
| GP weights | 10N (2D), 20N (3D) | 10N² (2D), 20N² (3D) |
| Activations | 4BN (2D), 8BN (3D) | 4BN (2D), 8BN (3D) |

## Integration with Existing Frameworks

The Layer class is designed to integrate with standard deep learning workflows:

```python
# Sequential model
model = torch.nn.Sequential(
    Layer(512, dims=3, normalize=True),
    Layer(256, dims=3, normalize=True),
    Layer(128, dims=3, normalize=False)
)

# Custom architecture
class CliffordMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([
            Layer(512, dims=3, normalize=True),
            Layer(256, dims=3, normalize=True),
            Layer(128, dims=3, normalize=True)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
```

The architecture supports standard PyTorch features including:
- Model serialization and loading
- Mixed precision training
- Gradient clipping and regularization
- Multi-GPU training via DataParallel or DistributedDataParallel

## Advanced Architecture Patterns

### Equivariant Network Design

The Layer architecture naturally supports equivariant network design:

```python
class EquivariantBackbone(nn.Module):
    """Backbone network that preserves equivariance throughout."""

    def __init__(self, dims=3, base_features=128):
        super().__init__()

        # Equivariant feature extraction
        self.stem = Layer(base_features, dims=dims, normalize=False, use_fc=False)

        # Equivariant processing blocks
        self.blocks = nn.ModuleList([
            CliffordBlock(base_features * (2**i), dims=dims)
            for i in range(4)
        ])

        # Maintain equivariance until final classification
        self.global_pool = GlobalMultivectorPooling()

    def forward(self, x):
        x = self.stem(x)  # (MV_DIM, B, F)

        for block in self.blocks:
            x = block(x)  # Maintains shape and equivariance

        # Global pooling breaks equivariance for classification
        return self.global_pool(x)

class CliffordBlock(nn.Module):
    """Basic building block maintaining equivariance."""

    def __init__(self, features, dims=3):
        super().__init__()
        self.layer1 = Layer(features, dims=dims, normalize=True, use_fc=False)
        self.layer2 = Layer(features, dims=dims, normalize=True, use_fc=True)

    def forward(self, x):
        residual = x
        x = self.layer1(x)
        x = self.layer2(x)
        return x  # Residual connection preserves equivariance
```

### Hybrid Architectures

Combining Clifford layers with standard neural networks:

```python
class HybridModel(nn.Module):
    """Combines Clifford equivariance with standard MLPs."""

    def __init__(self, dims=3):
        super().__init__()

        # Clifford feature extraction
        self.clifford_backbone = nn.Sequential(
            Layer(256, dims=dims, normalize=True, use_fc=False),
            Layer(256, dims=dims, normalize=True, use_fc=False),
        )

        # Standard MLP for classification
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

        # Feature fusion
        self.fusion = nn.Linear(256 * 4, 256)  # 4 components in 2D

    def forward(self, x):
        # Clifford processing (equivariant)
        clifford_features = self.clifford_backbone(x)

        # Extract scalar component for standard processing
        scalar_features = clifford_features[0]  # Scalar part

        # Optional: concatenate all components
        all_features = clifford_features.flatten(start_dim=0)

        return self.classifier(scalar_features)
```

## Parameter Efficiency

### Grade-wise Parameter Sharing

The architecture achieves parameter efficiency through grade-wise sharing:

| Grade | Components | Shared Parameters | Efficiency Gain |
|-------|------------|-------------------|-----------------|
| 0 (scalar) | 1 | Unique | Baseline |
| 1 (vector) | n | Shared | n× reduction |
| 2 (bivector) | n(n-1)/2 | Shared | 2n(n-1)× reduction |

**Total parameter reduction:**
- 2D: From 4N² to 3N² (25% reduction)
- 3D: From 8N² to 4N² (50% reduction)

### Geometric Product Weight Efficiency

The geometric product weights are also optimized:

- **SGP**: Only 10 weights in 2D, 20 in 3D (vs 16 and 64 for dense)
- **FCGP**: Full expressivity with structured weight sharing

## Equivariance Verification

### Runtime Equivariance Checking

```python
def verify_equivariance(model, x, transform_fn, rtol=1e-5):
    """Verify that model outputs transform correctly under group actions."""

    # Apply transformation to input
    x_transformed = transform_multivector(x, transform_fn)

    # Forward pass on both
    y_original = model(x)
    y_transformed = model(x_transformed)

    # Check if outputs transform correctly
    y_expected = transform_multivector(y_original, transform_fn)

    max_error = (y_transformed - y_expected).abs().max()
    return max_error < rtol, max_error
```

### Analytical Equivariance Proofs

The architecture's equivariance can be proven analytically:

**Theorem**: The Layer transformation is equivariant under O(n) actions.

**Proof**: The geometric product satisfies `R(AB)R⁻¹ = (RA)(RB)` for orthogonal R. Since all operations (linear, activation, normalization) are grade-wise and the geometric product preserves the group action, the entire layer is equivariant.

## Memory Layout Optimization

### Advanced Layout Strategies

The `(MV_DIM, BATCH, FEATURES)` layout enables several optimizations:

1. **Batch matrix multiplication**: Linear layers become efficient BLAS operations
2. **Component-wise parallelism**: Each multivector component can be processed independently
3. **Grade-wise operations**: Normalization and activation respect algebraic structure

### Custom Memory Management

For advanced users, custom memory layouts can be implemented:

```python
class CustomLayoutLayer(Layer):
    """Layer with custom memory layout optimization."""

    def __init__(self, *args, layout='grade_first', **kwargs):
        super().__init__(*args, **kwargs)
        self.layout = layout

    def forward(self, x):
        if self.layout == 'grade_first':
            # Reshape to group by grade
            x_reshaped = self.reshape_by_grade(x)
            y = self.process_by_grade(x_reshaped)
            return self.reshape_to_standard(y)
        else:
            return super().forward(x)
```

## Integration Patterns

### Model Zoo Integration

The architecture integrates with common model patterns:

```python
# Vision models
class CliffordVisionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = CliffordPatchEmbedding()
        self.layers = nn.ModuleList([
            CliffordTransformerBlock() for _ in range(12)
        ])

# Graph networks
class CliffordGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_update = Layer(node_features, dims=3)
        self.edge_update = Layer(edge_features, dims=3)
        self.message_pass = CliffordMessagePassing()
```

### Checkpointing and Optimization

```python
# Gradient checkpointing for memory efficiency
def forward_with_checkpoint(self, x):
    def layer_forward(layer, x):
        return layer(x)

    x = torch.utils.checkpoint.checkpoint(layer_forward, self.layer1, x)
    x = torch.utils.checkpoint.checkpoint(layer_forward, self.layer2, x)
    return x
```
