# Usage Examples

This section provides practical examples demonstrating how to use Flash Clifford for various geometric deep learning tasks.

## Basic Usage

### Simple Layer

```python
import torch
from modules.layer import Layer

# Create a 2D Clifford layer
layer = Layer(n_features=512, dims=2, normalize=True, use_fc=False)

# Input: multivectors in 2D of shape (4, batch_size, features)
x = torch.randn(4, 4096, 512).cuda()

# Forward pass
output = layer(x)
print(f"Input shape: {x.shape}, Output shape: {output.shape}")
```

### 3D Clifford Layer

```python
# Create a 3D Clifford layer with fully connected geometric product
layer_3d = Layer(n_features=256, dims=3, normalize=True, use_fc=True)

# Input: multivectors in 3D of shape (8, batch_size, features)
x = torch.randn(8, 2048, 256).cuda()

# Forward pass
output = layer_3d(x)
print(f"3D Input shape: {x.shape}, Output shape: {output.shape}")
```

## Building Neural Networks

### Sequential Model

```python
import torch.nn as nn
from modules.layer import Layer

class CliffordMLP(nn.Module):
    """Multi-layer perceptron using Clifford algebra layers."""

    def __init__(self, input_features=512, hidden_features=256, output_features=128, dims=3):
        super().__init__()

        self.layers = nn.Sequential(
            Layer(input_features, dims=dims, normalize=True, use_fc=False),
            Layer(hidden_features, dims=dims, normalize=True, use_fc=False),
            Layer(hidden_features, dims=dims, normalize=True, use_fc=False),
            Layer(output_features, dims=dims, normalize=False, use_fc=False)
        )

    def forward(self, x):
        return self.layers(x)

# Create and use the model
model = CliffordMLP(input_features=512, hidden_features=256, output_features=10, dims=3)
x = torch.randn(8, 4096, 512).cuda()
output = model(x)
print(f"Model output shape: {output.shape}")
```

### Residual Network

```python
class CliffordResNet(nn.Module):
    """Residual network with Clifford algebra layers."""

    def __init__(self, features=512, dims=3, num_layers=6):
        super().__init__()

        self.input_proj = Layer(features, dims=dims, normalize=False, use_fc=False)

        layers = []
        for i in range(num_layers):
            layers.append(
                CliffordResidualBlock(features, dims=dims, use_fc=(i % 2 == 1))
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_proj(x)
        return self.layers(x)

class CliffordResidualBlock(nn.Module):
    """Residual block for Clifford networks."""

    def __init__(self, features, dims=3, use_fc=False):
        super().__init__()

        self.layer1 = Layer(features, dims=dims, normalize=True, use_fc=use_fc)
        self.layer2 = Layer(features, dims=dims, normalize=False, use_fc=use_fc)

    def forward(self, x):
        residual = x
        x = self.layer1(x)
        x = self.layer2(x)
        return x  # Skip connection handled by input normalization

# Usage
model = CliffordResNet(features=512, dims=3, num_layers=6)
x = torch.randn(8, 4096, 512).cuda()
output = model(x)
```

## Direct Operation Usage

### Custom Operations

```python
import torch
from ops import fused_gelu_sgp_norm_2d, fused_gelu_fcgp_norm_3d

# 2D operations
x2d = torch.randn(4, 4096, 512).cuda()
y2d = torch.randn(4, 4096, 512).cuda()
weight2d = torch.randn(512, 10).cuda()

# Weighted geometric product
output_sgp = fused_gelu_sgp_norm_2d(x2d, y2d, weight2d, normalize=True)

# 3D operations
x3d = torch.randn(8, 2048, 256).cuda()
y3d = torch.randn(8, 2048, 256).cuda()
weight3d = torch.randn(256, 20).cuda()

# Weighted geometric product
output_sgp_3d = fused_gelu_sgp_norm_3d(x3d, y3d, weight3d, normalize=True)

# Fully connected geometric product
weight_fcgp = torch.randn(20, 256, 256).cuda()
output_fcgp = fused_gelu_fcgp_norm_3d(x3d, y3d, weight_fcgp, normalize=True)
```

## Training Examples

### Basic Training Loop

```python
import torch
import torch.nn as nn
from modules.layer import Layer

class SimpleCliffordModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, output_dim=10, dims=2):
        super().__init__()
        self.layer1 = Layer(input_dim, dims=dims, normalize=True, use_fc=False)
        self.layer2 = Layer(hidden_dim, dims=dims, normalize=True, use_fc=False)
        self.layer3 = Layer(output_dim, dims=dims, normalize=False, use_fc=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)

# Training setup
model = SimpleCliffordModel(input_dim=512, hidden_dim=256, output_dim=10, dims=2)
model.cuda()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

# Training data (example)
batch_size = 4096
num_features = 512
num_classes = 10

# Training loop
model.train()
for epoch in range(100):
    # Generate dummy data
    x = torch.randn(4, batch_size, num_features).cuda()  # 2D multivectors
    target = torch.randint(0, num_classes, (batch_size,)).cuda()

    # Forward pass
    output = model(x)
    # For classification, you might need to pool across multivector components
    logits = output[0]  # Use scalar component for classification
    loss = criterion(logits, target)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Equivariant Learning

```python
class EquivariantClassifier(nn.Module):
    """Classifier that respects O(n) equivariance."""

    def __init__(self, features=512, dims=3, num_classes=10):
        super().__init__()

        self.clifford_layers = nn.Sequential(
            Layer(features, dims=dims, normalize=True, use_fc=False),
            Layer(features, dims=dims, normalize=True, use_fc=False),
            Layer(features // 2, dims=dims, normalize=True, use_fc=False),
        )

        # Final classification layer (breaks equivariance)
        self.classifier = nn.Linear(features // 2, num_classes)

    def forward(self, x):
        # Clifford processing (equivariant)
        x = self.clifford_layers(x)

        # Classification (use scalar component)
        scalar_component = x[0]  # Scalar part
        return self.classifier(scalar_component.mean(dim=-1))  # Global average pooling

# Training with equivariance considerations
model = EquivariantClassifier(features=512, dims=3, num_classes=10)
model.cuda()

# Example: training on rotated data
def apply_rotation_3d(x, rotation_matrix):
    """Apply 3D rotation to multivector components."""
    # Vector components transform as vectors
    vectors = x[1:4]  # x, y, z components
    rotated_vectors = torch.matmul(rotation_matrix, vectors)

    # Bivectors transform as bivectors
    bivectors = x[4:7]  # xy, xz, yz components
    rotated_bivectors = torch.matmul(rotation_matrix, bivectors)

    return torch.cat([x[0:1], rotated_vectors, rotated_bivectors, x[7:8]], dim=0)

# Training with rotation augmentation
for batch in dataloader:
    x, target = batch
    x = x.cuda()

    # Apply random rotation
    rotation = random_rotation_3d()
    x_rotated = apply_rotation_3d(x, rotation)

    # Model should be equivariant to rotations
    output_original = model(x)
    output_rotated = model(x_rotated)

    loss = (criterion(output_original, target) + criterion(output_rotated, target)) / 2
    loss.backward()
```

## Performance Optimization

### Mixed Precision Training

```python
from torch.cuda.amp import GradScaler, autocast

# Enable mixed precision
scaler = GradScaler()

# Training with AMP
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        x, target = batch
        x = x.cuda()

        with autocast():
            output = model(x)
            loss = criterion(output, target)

        # Scale gradients and optimize
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### Gradient Accumulation

```python
# For very large batch sizes or limited memory
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    x, target = batch
    x = x.cuda()

    # Forward pass
    output = model(x)
    loss = criterion(output, target) / accumulation_steps

    # Backward pass (accumulated)
    loss.backward()

    # Update weights every N steps
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Benchmarking and Profiling

### Runtime Benchmarking

```python
from tests.utils import run_benchmark

# Compare your implementation with baseline
def custom_forward(x, y, weight):
    return model(x, y, weight)

# Benchmark setup
batch_size = 4096
num_features = 512
x = torch.randn(4, batch_size, num_features).cuda()
y = torch.randn(4, batch_size, num_features).cuda()
weight = torch.randn(num_features, 10).cuda()

# Run benchmark
result = run_benchmark(
    fused_gelu_sgp_norm_2d,
    gelu_sgp_norm_2d_torch,
    (x, y, weight),
    rep=1000,
    verbose=True
)
```

### Memory Profiling

```python
import torch.profiler as profiler

# Profile memory usage
with profiler.profile(
    activities=[profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    output = model(x)

print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
```

## Integration with Existing Frameworks

### Hugging Face Integration

```python
from transformers import Trainer, TrainingArguments

class CliffordModelForSequenceClassification:
    """Wrapper for Hugging Face integration."""

    def __init__(self, config):
        self.config = config
        self.clifford_model = CliffordMLP(
            input_features=config.hidden_size,
            hidden_features=config.hidden_size // 2,
            output_features=config.num_labels,
            dims=3
        )

    def forward(self, input_ids=None, multivector_input=None, labels=None):
        if multivector_input is not None:
            logits = self.clifford_model(multivector_input)
        else:
            # Convert input_ids to multivector representation
            multivector_input = self.embedding(input_ids)
            logits = self.clifford_model(multivector_input)

        loss = None
        if labels is not None:
            loss = criterion(logits, labels)

        return {'loss': loss, 'logits': logits}
```

## Advanced Usage Patterns

### Custom Geometric Products

```python
def custom_geometric_layer(x, y, weights):
    """Custom layer combining multiple geometric products."""

    # Multiple geometric products
    gp1 = fused_gelu_sgp_norm_2d(x, y, weights['gp1'], normalize=True)
    gp2 = fused_gelu_fcgp_norm_2d(x, y, weights['gp2'], normalize=False)

    # Combine products
    combined = torch.cat([gp1, gp2], dim=0)
    return fused_gelu_sgp_norm_2d(gp1, gp2, weights['combine'], normalize=True)
```

### Hierarchical Clifford Networks

```python
class HierarchicalClifford(nn.Module):
    """Network with multiple Clifford algebras."""

    def __init__(self):
        super().__init__()

        # 2D processing
        self.layer_2d = Layer(512, dims=2, normalize=True, use_fc=False)

        # 3D processing
        self.layer_3d = Layer(512, dims=3, normalize=True, use_fc=True)

        # Fusion layer
        self.fusion = Layer(1024, dims=3, normalize=False, use_fc=False)

    def forward(self, x_2d, x_3d):
        # Process each representation
        out_2d = self.layer_2d(x_2d)
        out_3d = self.layer_3d(x_3d)

        # Fuse representations
        combined = torch.cat([out_2d, out_3d], dim=0)
        return self.fusion(combined)
```

## Best Practices

### Memory Management

```python
# Use appropriate tensor types
x = torch.randn(4, 4096, 512, dtype=torch.float32).cuda()

# Ensure contiguous memory layout
x = x.contiguous()

# Use gradient checkpointing for large models
from torch.utils.checkpoint import checkpoint
output = checkpoint(model.layer1, x)
```

### Numerical Stability

```python
# Use appropriate normalization
layer = Layer(features, dims=dims, normalize=True)  # Enable normalization

# Monitor gradient norms
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** (1. / 2)

# Gradient clipping if needed
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Performance Monitoring

```python
import time

# Time forward pass
start_time = time.time()
for _ in range(100):
    output = model(x)
torch.cuda.synchronize()
end_time = time.time()

print(f"Average forward time: {(end_time - start_time) / 100 * 1000:.2f}ms")

# Monitor memory usage
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

These examples demonstrate the flexibility and performance of Flash Clifford for various geometric deep learning applications, from simple classification to complex equivariant architectures.

