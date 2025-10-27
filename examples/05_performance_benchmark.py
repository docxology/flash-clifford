#!/usr/bin/env python3
"""
Performance Benchmark Example for Flash Clifford

This example demonstrates performance characteristics of Flash Clifford
operations and provides benchmarking utilities.

Output: Saves results to output/performance_benchmark/
"""

import torch
import time
import os

# Create output directory
output_dir = "output/performance_benchmark"
os.makedirs(output_dir, exist_ok=True)

print("⚡ Flash Clifford - Performance Benchmark Example")
print("=" * 50)

def benchmark_operation(operation_name, func, *args, warmup=10, iterations=100):
    """Benchmark a single operation."""
    print(f"Benchmarking {operation_name}...")

    # Warmup
    for _ in range(warmup):
        _ = func(*args)

    # Timed runs
    start_time = time.time()
    for _ in range(iterations):
        result = func(*args)
        # Prevent optimization from eliminating computation
        result.sum().item()
    end_time = time.time()

    avg_time_ms = (end_time - start_time) / iterations * 1000

    # Calculate throughput
    if len(args) > 0:
        total_elements = sum(arg.numel() for arg in args)
        throughput = total_elements * iterations / (end_time - start_time) / 1e6  # Melements/sec
    else:
        throughput = iterations / (end_time - start_time)  # operations/sec

    print(f"  ✅ {operation_name}: {avg_time_ms:.3f} ms, {throughput:.1f} Melements/sec")

    return {
        'operation': operation_name,
        'avg_time_ms': avg_time_ms,
        'throughput_melements_sec': throughput,
        'iterations': iterations
    }

try:
    # Import baseline operations
    import importlib.util
    spec = importlib.util.spec_from_file_location("baselines", "tests/baselines.py")
    baselines = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baselines)

    print("✅ Successfully imported baseline operations")

    # Test different configurations
    configurations = [
        (4, 64, 128),   # Small 2D
        (4, 256, 512),  # Medium 2D
        (4, 1024, 1024), # Large 2D
    ]

    all_results = []

    for mv_dim, batch_size, n_features in configurations:
        print(f"\n📊 Configuration: {mv_dim}D, {batch_size}×{n_features}")
        print("-" * 40)

        # Create test data
        x = torch.randn(mv_dim, batch_size, n_features)
        y = torch.randn(mv_dim, batch_size, n_features)

        if mv_dim == 4:  # 2D
            weight = torch.randn(n_features, 10)
        else:  # 3D
            weight = torch.randn(n_features, 20)

        # Benchmark individual operations
        operations = [
            ("GELU Activation", lambda: baselines.mv_gelu(x)),
            ("RMS Normalization", lambda: baselines.mv_rmsnorm_2d(x) if mv_dim == 4 else baselines.mv_rmsnorm_3d(x.reshape(8, batch_size, n_features))),
        ]

        config_results = []

        for op_name, op_func in operations:
            result = benchmark_operation(op_name, op_func, iterations=50)
            config_results.append(result)

        # Test memory usage
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Run operations multiple times
        for _ in range(10):
            _ = baselines.mv_gelu(x)
            if mv_dim == 4:
                _ = baselines.mv_rmsnorm_2d(x)
                _ = baselines.sgp_2d(x, y, weight)

        final_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memory_increase = (final_memory - initial_memory) / 1e6 if torch.cuda.is_available() else 0

        print(f"  💾 Memory usage: {memory_increase:.1f} MB increase")

        config_results.append({
            'operation': 'Memory Usage',
            'memory_increase_mb': memory_increase,
            'config': f'{mv_dim}D_{batch_size}_{n_features}'
        })

        all_results.extend(config_results)

    # Summary
    print("
📈 Performance Summary"    print("-" * 20)

    for result in all_results:
        if 'avg_time_ms' in result:
            print(f"{result['operation']"20"} | {result['avg_time_ms']"8.3f"} ms | {result['throughput_melements_sec']"10.1f"} Melem/s")

    # Save detailed results
    torch.save({
        'benchmark_results': all_results,
        'configurations_tested': configurations,
        'hardware_info': {
            'cuda_available': torch.cuda.is_available(),
            'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
            'num_cuda_devices': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
        'performance_metrics': {
            'total_operations_tested': len([r for r in all_results if 'avg_time_ms' in r]),
            'fastest_operation': min([r['avg_time_ms'] for r in all_results if 'avg_time_ms' in r]),
            'slowest_operation': max([r['avg_time_ms'] for r in all_results if 'avg_time_ms' in r]),
        }
    }, f"{output_dir}/performance_benchmark_results.pt")

    print(f"\n💾 Detailed results saved to: {output_dir}/performance_benchmark_results.pt")

    print("\n🎉 Performance benchmark example completed successfully!")
    print("✅ Performance characteristics measured")
    print("✅ Memory usage tracked")
    print("✅ Multiple configurations tested")

except Exception as e:
    print(f"❌ Error during execution: {e}")
    raise

print("\n" + "=" * 50)
print("Example completed. Check output/performance_benchmark/ for results.")
