import os

import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def benchmark_forward(fn, args, n_measure, warmup=50):
    """Benchmark forward pass with warmup."""
    for _ in range(warmup):
        _ = fn(*args)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    total_time = 0
    
    for _ in range(n_measure):
        torch.cuda.synchronize()
        start.record()
        _ = fn(*args)
        end.record()
        torch.cuda.synchronize()
        total_time += start.elapsed_time(end)
    
    return total_time / n_measure


def benchmark_backward_triton(fn, x, y, weight, n_measure, warmup=50, fn_name="Triton"):
    """Benchmark forward + backward for Triton kernels."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Get output shape for creating gradient tensors
    try:
        with torch.no_grad():
            out_example = fn(x, y, weight)
        grad_shape = out_example.shape
        del out_example
    except RuntimeError as e:
        if "out of memory" in str(e):
            if fn_name:
                print(f"  OOM in {fn_name} (pre-allocation)")
            torch.cuda.empty_cache()
            return None
        raise

    # Warmup
    for _ in range(warmup):
        try:
            x_grad = x.clone().requires_grad_(True)
            y_grad = y.clone().requires_grad_(True)
            weight_grad = weight.clone().requires_grad_(True)
            out = fn(x_grad, y_grad, weight_grad)
            grad_output_warmup = torch.randn(grad_shape, device=x.device, dtype=x.dtype)
            out.backward(grad_output_warmup)
            del x_grad, y_grad, weight_grad, out, grad_output_warmup
        except RuntimeError as e:
            if "out of memory" in str(e):
                if fn_name:
                    print(f"  OOM in {fn_name} (warmup)")
                torch.cuda.empty_cache()
                return None
            raise

    # Benchmark
    total_time = 0
    for i in range(n_measure):
        x_grad = y_grad = weight_grad = out = grad_output = None

        try:
            x_grad = x.clone().requires_grad_(True)
            y_grad = y.clone().requires_grad_(True)
            weight_grad = weight.clone().requires_grad_(True)
            grad_output = torch.randn(grad_shape, device=x.device, dtype=x.dtype)

            torch.cuda.synchronize()
            start.record()
            out = fn(x_grad, y_grad, weight_grad)
            out.backward(grad_output)
            end.record()
            torch.cuda.synchronize()
            total_time += start.elapsed_time(end)

            del x_grad, y_grad, weight_grad, out, grad_output
        except RuntimeError as e:
            if "out of memory" in str(e):
                if fn_name:
                    print(f"  OOM in {fn_name} (iteration {i})")
                for var in [x_grad, y_grad, weight_grad, out, grad_output]:
                    if var is not None:
                        del var
                torch.cuda.empty_cache()
                return None
            raise

    return total_time / n_measure


def benchmark_backward_torch(fn, x, y, sgp, n_measure, warmup=50, fn_name="PyTorch"):
    """Benchmark forward + backward for PyTorch implementations."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Get output shape for creating gradient tensors
    try:
        with torch.no_grad():
            out_example = fn(x, y, sgp._get_weight())
        grad_shape = out_example.shape
        del out_example
    except RuntimeError as e:
        if "out of memory" in str(e):
            if fn_name:
                print(f"  OOM in {fn_name} (pre-allocation)")
            torch.cuda.empty_cache()
            return None
        raise

    # Warmup
    for _ in range(warmup):
        try:
            x_grad = x.clone().requires_grad_(True)
            y_grad = y.clone().requires_grad_(True)
            out = fn(x_grad, y_grad, sgp._get_weight())
            grad_output_warmup = torch.randn(grad_shape, device=x.device, dtype=x.dtype)
            out.backward(grad_output_warmup)
            del x_grad, y_grad, out, grad_output_warmup
        except RuntimeError as e:
            if "out of memory" in str(e):
                if fn_name:
                    print(f"  OOM in {fn_name} (warmup)")
                torch.cuda.empty_cache()
                return None
            raise

    # Benchmark
    total_time = 0
    for i in range(n_measure):
        x_grad = y_grad = out = grad_output = None

        try:
            x_grad = x.clone().requires_grad_(True)
            y_grad = y.clone().requires_grad_(True)
            grad_output = torch.randn(grad_shape, device=x.device, dtype=x.dtype)

            torch.cuda.synchronize()
            start.record()
            out = fn(x_grad, y_grad, sgp._get_weight())
            out.backward(grad_output)
            end.record()
            torch.cuda.synchronize()
            total_time += start.elapsed_time(end)

            del x_grad, y_grad, out, grad_output
        except RuntimeError as e:
            if "out of memory" in str(e):
                if fn_name:
                    print(f"  OOM in {fn_name} (iteration {i})")
                for var in [x_grad, y_grad, out, grad_output]:
                    if var is not None:
                        del var
                torch.cuda.empty_cache()
                return None
            raise

    return total_time / n_measure


def measure_memory_forward(fn, *args):
    """Measure peak memory usage for forward pass in MB."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    try:
        with torch.no_grad():
            _ = fn(*args)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024**2
    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            return None
        raise


def measure_memory_backward_triton(fn, x, y, weight, fn_name="Triton"):
    """Measure peak memory usage for Triton kernels."""
    x_grad = y_grad = weight_grad = out = grad_output = None

    try:
        torch.cuda.reset_peak_memory_stats()

        x_grad = x.clone().detach().requires_grad_(True)
        y_grad = y.clone().detach().requires_grad_(True)
        weight_grad = weight.clone().detach().requires_grad_(True)

        torch.cuda.synchronize()
        out = fn(x_grad, y_grad, weight_grad)
        grad_output = torch.randn_like(out)
        out.backward(grad_output)
        torch.cuda.synchronize()

        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        del x_grad, y_grad, weight_grad, out, grad_output
        return peak_memory
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            if fn_name:
                print(f"  OOM in {fn_name}")
            for var in [x_grad, y_grad, weight_grad, out, grad_output]:
                if var is not None:
                    del var
            torch.cuda.empty_cache()
            return None
        raise


def measure_memory_backward_torch(fn, x, y, sgp, fn_name="PyTorch"):
    """Measure peak memory usage for PyTorch implementations."""
    x_grad = y_grad = out = grad_output = None

    try:
        torch.cuda.reset_peak_memory_stats()

        x_grad = x.clone().detach().requires_grad_(True)
        y_grad = y.clone().detach().requires_grad_(True)

        torch.cuda.synchronize()
        out = fn(x_grad, y_grad, sgp._get_weight())
        grad_output = torch.randn_like(out)
        out.backward(grad_output)
        torch.cuda.synchronize()

        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        del x_grad, y_grad, out, grad_output
        return peak_memory
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            if fn_name:
                print(f"  OOM in {fn_name}")
            for var in [x_grad, y_grad, out, grad_output]:
                if var is not None:
                    del var
            torch.cuda.empty_cache()
            return None
        raise


def plot_heatmap(results, metric_key, title, save_path, cmap='RdYlGn', invert_cmap=False):
    """Heatmap visualization of benchmark results."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    batch_sizes = sorted(set(r['batch_size'] for r in results))
    num_features_list = sorted(set(r['num_features'] for r in results))

    matrix = np.zeros((len(batch_sizes), len(num_features_list)))
    for r in results:
        i = batch_sizes.index(r['batch_size'])
        j = num_features_list.index(r['num_features'])
        matrix[i, j] = r[metric_key] if r[metric_key] is not None else 0

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap_name = f'{cmap}_r' if invert_cmap else cmap
    im = ax.imshow(matrix, cmap=cmap_name, aspect='auto')

    ax.set_xticks(np.arange(len(num_features_list)))
    ax.set_yticks(np.arange(len(batch_sizes)))
    ax.set_xticklabels(num_features_list)
    ax.set_yticklabels(batch_sizes)

    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Batch Size', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)

    cbar = plt.colorbar(im, ax=ax)
    cbar_label = 'Speedup (x)' if 'speedup' in metric_key else 'Memory Ratio (x)'
    cbar.set_label(cbar_label, fontsize=12)

    # Add text annotations
    for i in range(len(batch_sizes)):
        for j in range(len(num_features_list)):
            value = matrix[i, j]
            text_str = f'{value:.2f}x' if value > 0 else 'OOM'
            ax.text(j, i, text_str, ha="center", va="center", color="black", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to: {save_path}")
    plt.close()


def run_correctness_tests(triton_fn, torch_fn, x, y, weight, sgp):
    """Run forward and backward correctness tests."""
    # Forward correctness check
    out_triton = triton_fn(x, y, weight)
    out_torch = torch_fn(x, y, sgp._get_weight())

    max_diff = (out_torch - out_triton).abs().max().item()
    is_correct = torch.allclose(out_torch, out_triton, atol=1e-5)
    check_mark = " ✔" if is_correct else " ✘"
    print(f"Max absolute difference (fwd): {max_diff:.1e}{check_mark}")

    # Backward correctness check
    x_torch = x.clone().detach().requires_grad_(True)
    y_torch = y.clone().detach().requires_grad_(True)

    x_triton = x.clone().detach().requires_grad_(True)
    y_triton = y.clone().detach().requires_grad_(True)
    weight_triton = weight.detach().clone().requires_grad_(True)

    out_torch = torch_fn(x_torch, y_torch, sgp._get_weight())
    out_triton = triton_fn(x_triton, y_triton, weight_triton)

    grad_output = torch.randn_like(out_torch)
    out_torch.backward(grad_output)
    out_triton.backward(grad_output)

    print("\nBackward correctness:")
    grad_x_diff = (x_torch.grad - x_triton.grad).abs().max().item()
    grad_x_correct = torch.allclose(x_torch.grad, x_triton.grad, atol=1e-1)
    print(f"grad_x max diff: {grad_x_diff:.1e}{' ✔' if grad_x_correct else ' ✘'}")
    
    grad_y_diff = (y_torch.grad - y_triton.grad).abs().max().item()
    grad_y_correct = torch.allclose(y_torch.grad, y_triton.grad, atol=1e-1)
    print(f"grad_y max diff: {grad_y_diff:.1e}{' ✔' if grad_y_correct else ' ✘'}")

    return sgp, weight_triton


def run_benchmarks(triton_fn, torch_fn, x, y, weight, sgp, n_measure, warmup_iters=5):
    """Run forward and forward+backward benchmarks."""
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = triton_fn(x, y, weight)
            _ = torch_fn(x, y, sgp._get_weight())

    # Forward-only benchmark
    avg_time_fused = benchmark_forward(triton_fn, (x, y, weight), n_measure, warmup=0)
    avg_time_torch = benchmark_forward(torch_fn, (x, y, sgp._get_weight()), n_measure, warmup=0)

    print(f"\nFwd time (fused kernel): {avg_time_fused:.2f} ms")
    print(f"Fwd time (torch): {avg_time_torch:.2f} ms")
    print(f"Fwd Speedup: {avg_time_torch / avg_time_fused:.2f}x")

    # Forward + backward benchmark
    avg_time_fused_bwd = benchmark_backward_triton(triton_fn, x, y, weight, n_measure, warmup=5)
    avg_time_torch_bwd = benchmark_backward_torch(torch_fn, x, y, sgp, n_measure, warmup=5)

    print(f"\nFwd + bwd time (fused kernel): {avg_time_fused_bwd:.2f} ms")
    print(f"Fwd + bwd time (torch): {avg_time_torch_bwd:.2f} ms")
    print(f"Fwd + bwd Speedup: {avg_time_torch_bwd / avg_time_fused_bwd:.2f}x")


def print_results_table(results, title):
    """Print results as a formatted table."""
    separator = "=" * 140
    divider = "-" * 140

    # Forward pass results
    print(f"\n{separator}")
    print(f"FORWARD PASS RESULTS - {title}")
    print(separator)
    print(f"{'Batch':<8} {'Features':<10} {'Fused (ms)':<12} {'Torch (ms)':<12} "
          f"{'Speedup':<10} {'Max Diff':<12} {'Correct':<8}")
    print(divider)

    for r in results:
        speedup_str = f"{r['speedup_fwd']:.2f}x" if r['speedup_fwd'] else "N/A"
        correct_mark = '✔' if r['is_correct'] else '✘'
        print(f"{r['batch_size']:<8} {r['num_features']:<10} {r['time_fwd_fused']:<12.2f} "
              f"{r['time_fwd_torch']:<12.2f} {speedup_str:<10} {r['max_diff']:<12.1e} "
              f"{correct_mark:<8}")

    print(separator)

    # Forward + backward pass results
    print(f"\n{separator}")
    print(f"FORWARD + BACKWARD PASS RESULTS - {title}")
    print(separator)
    print(f"{'Batch':<8} {'Features':<10} {'Fused (ms)':<12} {'Torch (ms)':<12} {'Speedup':<10}")
    print(divider)

    for r in results:
        fused_time = f"{r['time_fwd_bwd_fused']:.2f}" if r['time_fwd_bwd_fused'] else "OOM"
        torch_time = f"{r['time_fwd_bwd_torch']:.2f}" if r['time_fwd_bwd_torch'] else "OOM"
        speedup = f"{r['speedup_fwd_bwd']:.2f}x" if r['speedup_fwd_bwd'] else "N/A"
        print(f"{r['batch_size']:<8} {r['num_features']:<10} {fused_time:<12} "
              f"{torch_time:<12} {speedup:<10}")

    print(separator)

    # Forward memory usage
    print(f"\n{separator}")
    print(f"FORWARD MEMORY USAGE - {title}")
    print(separator)
    print(f"{'Batch':<8} {'Features':<10} {'Fused (MB)':<12} {'Torch (MB)':<12} {'Ratio':<10}")
    print(divider)

    for r in results:
        fused_mem = f"{r['mem_fwd_fused']:.2f}" if r['mem_fwd_fused'] else "OOM"
        torch_mem = f"{r['mem_fwd_torch']:.2f}" if r['mem_fwd_torch'] else "OOM"
        ratio = f"{r['mem_ratio_fwd']:.2f}x" if r['mem_ratio_fwd'] else "N/A"
        print(f"{r['batch_size']:<8} {r['num_features']:<10} {fused_mem:<12} "
              f"{torch_mem:<12} {ratio:<10}")

    print(separator)

    # Forward + backward memory usage
    print(f"\n{separator}")
    print(f"FORWARD + BACKWARD MEMORY USAGE - {title}")
    print(separator)
    print(f"{'Batch':<8} {'Features':<10} {'Fused (MB)':<12} {'Torch (MB)':<12} {'Ratio':<10}")
    print(divider)

    for r in results:
        fused_mem = f"{r['mem_fwd_bwd_fused']:.2f}" if r['mem_fwd_bwd_fused'] else "OOM"
        torch_mem = f"{r['mem_fwd_bwd_torch']:.2f}" if r['mem_fwd_bwd_torch'] else "OOM"
        ratio = f"{r['mem_ratio_fwd_bwd']:.2f}x" if r['mem_ratio_fwd_bwd'] else "N/A"
        print(f"{r['batch_size']:<8} {r['num_features']:<10} {fused_mem:<12} "
              f"{torch_mem:<12} {ratio:<10}")

    print(separator)


def run_single_benchmark(triton_fn, torch_fn, setup_fn, batch_size=4096, 
                        num_features=512, n_measure=1000):
    """Run a single benchmark configuration."""
    x, y, weight, weight_expanded, sgp = setup_fn(batch_size, num_features)

    with torch.no_grad():
        out_triton = triton_fn(x, y, weight)
        out_torch = torch_fn(x, y, weight_expanded)

    is_correct = torch.allclose(out_torch, out_triton, atol=1e-5)
    max_diff = (out_torch - out_triton).abs().max().item()

    # Forward benchmarks
    time_fwd_fused = benchmark_forward(triton_fn, (x, y, weight), n_measure)
    time_fwd_torch = benchmark_forward(torch_fn, (x, y, weight_expanded), n_measure)

    # Forward + backward benchmarks
    time_fwd_bwd_fused = benchmark_backward_triton(triton_fn, x, y, weight, n_measure)
    time_fwd_bwd_torch = benchmark_backward_torch(torch_fn, x, y, sgp, n_measure)

    # Forward memory
    mem_fwd_fused = measure_memory_forward(triton_fn, x, y, weight)
    mem_fwd_torch = measure_memory_forward(torch_fn, x, y, weight_expanded)

    # Forward + backward memory
    mem_fwd_bwd_fused = measure_memory_backward_triton(triton_fn, x, y, weight)
    mem_fwd_bwd_torch = measure_memory_backward_torch(torch_fn, x, y, sgp)

    # Calculate speedup and ratios
    speedup_fwd = (time_fwd_torch / time_fwd_fused 
                   if time_fwd_fused and time_fwd_torch else None)
    speedup_fwd_bwd = (time_fwd_bwd_torch / time_fwd_bwd_fused 
                       if time_fwd_bwd_fused and time_fwd_bwd_torch else None)
    mem_ratio_fwd = (mem_fwd_fused / mem_fwd_torch 
                     if mem_fwd_fused and mem_fwd_torch and mem_fwd_torch > 0 else None)
    mem_ratio_fwd_bwd = (mem_fwd_bwd_fused / mem_fwd_bwd_torch 
                         if mem_fwd_bwd_fused and mem_fwd_bwd_torch and mem_fwd_bwd_torch > 0 
                         else None)

    return {
        'batch_size': batch_size,
        'num_features': num_features,
        'time_fwd_fused': time_fwd_fused,
        'time_fwd_torch': time_fwd_torch,
        'speedup_fwd': speedup_fwd,
        'time_fwd_bwd_fused': time_fwd_bwd_fused,
        'time_fwd_bwd_torch': time_fwd_bwd_torch,
        'speedup_fwd_bwd': speedup_fwd_bwd,
        'mem_fwd_fused': mem_fwd_fused,
        'mem_fwd_torch': mem_fwd_torch,
        'mem_ratio_fwd': mem_ratio_fwd,
        'mem_fwd_bwd_fused': mem_fwd_bwd_fused,
        'mem_fwd_bwd_torch': mem_fwd_bwd_torch,
        'mem_ratio_fwd_bwd': mem_ratio_fwd_bwd,
        'max_diff': max_diff,
        'is_correct': is_correct,
    }


def run_sweep(triton_fn, torch_fn, setup_fn,
              batch_sizes=[1024, 2048, 4096, 8192],
              num_features_list=[128, 256, 512, 1024],
              n_measure=1000):
    """Run benchmark sweep across batch sizes and feature dimensions."""
    results = []

    print("Running benchmark sweep...")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Num features: {num_features_list}")
    print(f"Measurements per config: {n_measure}\n")

    for batch_size in batch_sizes:
        for num_features in num_features_list:
            print(f"Running batch_size={batch_size}, num_features={num_features}...", end=" ")
            result = run_single_benchmark(triton_fn, torch_fn, setup_fn, 
                                         batch_size, num_features, n_measure)
            results.append(result)

            fwd_msg = (f"Fwd: {result['speedup_fwd']:.2f}x" 
                      if result['speedup_fwd'] else "Fwd: OOM")
            bwd_msg = (f"Fwd+Bwd: {result['speedup_fwd_bwd']:.2f}x" 
                      if result['speedup_fwd_bwd'] else "Fwd+Bwd: OOM")
            print(f"{fwd_msg}, {bwd_msg}")

    return results