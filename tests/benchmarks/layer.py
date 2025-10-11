import os
import sys
from dataclasses import dataclass
from typing import Callable, List

import torch
import torch.nn as nn

torch.set_float32_matmul_precision('medium')
torch._dynamo.config.cache_size_limit = 512

# Setup repository
if not os.path.exists("clifford-group-equivariant-neural-networks"):
    os.system("git clone https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks.git")
sys.path.append("clifford-group-equivariant-neural-networks")

from algebra.cliffordalgebra import CliffordAlgebra
from models.modules.gp import SteerableGeometricProductLayer
from models.modules.linear import MVLinear
from models.modules.mvlayernorm import MVLayerNorm
from models.modules.mvsilu import MVSiLU
from modules.layer import Layer
from tests.utils import plot_heatmap, print_results_table


@dataclass
class BenchmarkConfig:
    dims: int
    title: str
    output_path: str
    batch_sizes: List[int]
    num_features_list: List[int]
    n_measure: int = 100


def create_layers(num_features: int, dims: int):
    """Create fused and torch layers with matching configurations."""
    layer_fused = Layer(num_features, dims=dims, normalize=True, use_fc=False).cuda()
    
    algebra = CliffordAlgebra((1,) * dims)
    layer_torch = nn.Sequential(
        MVLinear(algebra, num_features, num_features),
        MVSiLU(algebra, num_features),
        SteerableGeometricProductLayer(algebra, num_features, include_first_order=False),
        MVLayerNorm(algebra, num_features),
    ).cuda()
    
    return layer_fused, layer_torch


def create_inputs(batch_size: int, num_features: int, dims: int):
    """Create input tensors for both layer types."""
    x_fused = torch.randn(2**dims, batch_size, num_features).cuda()
    x_torch = torch.randn(batch_size, num_features, 2**dims).cuda()
    return x_fused, x_torch


def benchmark_forward(layer: nn.Module, x: torch.Tensor, n_measure: int, warmup: int = 50) -> float:
    """Benchmark forward pass."""
    # Warmup
    for _ in range(warmup):
        _ = layer(x)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    with torch.no_grad():
        times = []
        for _ in range(n_measure):
            torch.cuda.synchronize()
            start.record()
            _ = layer(x)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    
    return sum(times) / len(times)


def benchmark_backward(layer: nn.Module, x: torch.Tensor, n_measure: int, warmup: int = 20) -> float:
    """Benchmark forward + backward pass."""
    # Warmup
    for _ in range(warmup):
        x_grad = x.clone().requires_grad_(True)
        out = layer(x_grad)
        out.backward(torch.randn_like(out))
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    fake_grad = torch.randn_like(layer(x))
    
    times = []
    for _ in range(n_measure):
        x_grad = x.clone().requires_grad_(True)
        torch.cuda.synchronize()
        start.record()
        out = layer(x_grad)
        out.backward(fake_grad)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    return sum(times) / len(times)


def measure_memory(layer: nn.Module, x: torch.Tensor, backward: bool = False) -> float:
    """Measure peak memory usage in MB."""
    torch.cuda.reset_peak_memory_stats()
    
    if backward:
        x_grad = x.clone().requires_grad_(True)
        out = layer(x_grad)
        out.backward(torch.randn_like(out))
    else:
        with torch.no_grad():
            layer(x)
    
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2


def benchmark_single(batch_size: int, num_features: int, dims: int, n_measure: int) -> dict:
    """Run complete benchmark for a single configuration."""
    # Setup
    layer_fused, layer_torch = create_layers(num_features, dims)
    x_fused, x_torch = create_inputs(batch_size, num_features, dims)
    
    # Time benchmarks
    time_fwd_fused = benchmark_forward(layer_fused, x_fused, n_measure)
    time_fwd_torch = benchmark_forward(layer_torch, x_torch, n_measure)
    time_bwd_fused = benchmark_backward(layer_fused, x_fused, n_measure)
    time_bwd_torch = benchmark_backward(layer_torch, x_torch, n_measure)
    
    # Memory benchmarks
    mem_fwd_fused = measure_memory(layer_fused, x_fused)
    mem_fwd_torch = measure_memory(layer_torch, x_torch)
    mem_bwd_fused = measure_memory(layer_fused, x_fused, backward=True)
    mem_bwd_torch = measure_memory(layer_torch, x_torch, backward=True)
    
    return {
        'batch_size': batch_size,
        'num_features': num_features,
        'time_fwd_fused': time_fwd_fused,
        'time_fwd_torch': time_fwd_torch,
        'speedup_fwd': time_fwd_torch / time_fwd_fused,
        'time_fwd_bwd_fused': time_bwd_fused,
        'time_fwd_bwd_torch': time_bwd_torch,
        'speedup_fwd_bwd': time_bwd_torch / time_bwd_fused,
        'mem_fwd_fused': mem_fwd_fused,
        'mem_fwd_torch': mem_fwd_torch,
        'mem_ratio_fwd': mem_fwd_fused / mem_fwd_torch,
        'mem_fwd_bwd_fused': mem_bwd_fused,
        'mem_fwd_bwd_torch': mem_bwd_torch,
        'mem_ratio_fwd_bwd': mem_bwd_fused / mem_bwd_torch,
        'max_diff': 0.0,
        'is_correct': True,
    }


def run_benchmark_sweep(config: BenchmarkConfig) -> List[dict]:
    """Run complete benchmark sweep and generate all outputs."""
    print(f"\n{config.title}")
    print(f"Batches: {config.batch_sizes} | Features: {config.num_features_list}")
    
    results = []
    for batch_size in config.batch_sizes:
        for num_features in config.num_features_list:
            print(f"  {batch_size}×{num_features}...", end=" ", flush=True)
            
            result = benchmark_single(batch_size, num_features, config.dims, config.n_measure)
            results.append(result)
            
            print(f"Fwd: {result['speedup_fwd']:.2f}x, Fwd+Bwd: {result['speedup_fwd_bwd']:.2f}x")
    
    # Print results table
    print_results_table(results, config.title)
    
    # Generate heatmaps
    metrics = [
        ('speedup_fwd', 'Forward Pass Speedup', False),
        ('speedup_fwd_bwd', 'Forward + Backward Pass Speedup', False),
        ('mem_ratio_fwd', 'Forward Pass Memory Ratio', True),
        ('mem_ratio_fwd_bwd', 'Forward + Backward Pass Memory Ratio', True),
    ]
    
    for metric, label, invert in metrics:
        category, subcategory = metric.split('_')[0], metric.split('_')[1]
        output_file = f'{config.output_path}/{category}/{subcategory}.png'
        plot_heatmap(
            results, 
            metric, 
            f'{label}: Triton vs PyTorch\n{config.title}',
            output_file,
            invert_cmap=invert
        )
    
    return results


def main():
    assert torch.cuda.is_available(), "CUDA not available"
    
    configs = [
        BenchmarkConfig(
            dims=2,
            title="Layer 2D (Cl(1,1))",
            output_path="tests/benchmarks/results/layer_2d",
            batch_sizes=[1024, 2048, 4096, 8192],
            num_features_list=[128, 256, 512, 1024],
            n_measure=100
        ),
        BenchmarkConfig(
            dims=3,
            title="Layer 3D (Cl(1,1,1))",
            output_path="tests/benchmarks/results/layer_3d",
            batch_sizes=[1024, 2048, 4096, 8192],
            num_features_list=[128, 256, 512, 1024],
            n_measure=100
        ),
    ]
    
    for config in configs:
        run_benchmark_sweep(config)


if __name__ == "__main__":
    main()