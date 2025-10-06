import torch


def mv_gelu(x):
    """Apply GELU activation gated by scalar component."""
    scalar = x[..., [0]]
    gate = 0.5 * (1 + torch.erf(scalar * 0.7071067811865475))
    return x * gate


def mv_rmsnorm_2d(x, eps=1e-6):
    """RMS normalization for 2D Clifford algebra (scalar, vector, pseudoscalar)."""
    scalar = x[..., [0]]
    vector = x[..., [1, 2]]
    pseudoscalar = x[..., [3]]

    scalar_rms = (scalar.pow(2).mean(dim=1, keepdim=True) + eps).sqrt()
    scalar = scalar / scalar_rms

    vector_norm = vector.norm(dim=2, keepdim=True)
    vector_rms = (vector_norm.pow(2).mean(dim=1, keepdim=True) + eps).sqrt()
    vector = vector / vector_rms

    pseudoscalar_rms = (pseudoscalar.pow(2).mean(dim=1, keepdim=True) + eps).sqrt()
    pseudoscalar = pseudoscalar / pseudoscalar_rms

    return torch.cat([scalar, vector, pseudoscalar], dim=-1)


def mv_rmsnorm_3d(x, eps=1e-6):
    """RMS normalization for 3D Clifford algebra (scalar, vector, bivector, pseudoscalar)."""
    scalar = x[..., [0]]
    vector = x[..., [1, 2, 3]]
    bivector = x[..., [4, 5, 6]]
    pseudoscalar = x[..., [7]]

    scalar_rms = (scalar.pow(2).mean(dim=1, keepdim=True) + eps).sqrt()
    scalar = scalar / scalar_rms

    vector_norm = vector.norm(dim=2, keepdim=True)
    vector_rms = (vector_norm.pow(2).mean(dim=1, keepdim=True) + eps).sqrt()
    vector = vector / vector_rms

    bivector_norm = bivector.norm(dim=2, keepdim=True)
    bivector_rms = (bivector_norm.pow(2).mean(dim=1, keepdim=True) + eps).sqrt()
    bivector = bivector / bivector_rms

    pseudoscalar_rms = (pseudoscalar.pow(2).mean(dim=1, keepdim=True) + eps).sqrt()
    pseudoscalar = pseudoscalar / pseudoscalar_rms

    return torch.cat([scalar, vector, bivector, pseudoscalar], dim=-1)


@torch.no_grad()
def benchmark_forward(fn, args, n_measure):
    """Benchmark forward pass only."""
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


def benchmark_backward_triton(fn, x, y, weight, n_measure):
    """Benchmark forward + backward pass for Triton implementation."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        out_example = fn(x, y, weight)
        grad_shape = out_example.shape

    total_time = 0
    for _ in range(n_measure):
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
    return total_time / n_measure


def benchmark_backward_torch(fn, x, y, sgp, n_measure):
    """Benchmark forward + backward pass for PyTorch implementation."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Get output shape for creating gradient tensors
    with torch.no_grad():
        out_example = fn(x, y, sgp)
        grad_shape = out_example.shape

    total_time = 0
    for _ in range(n_measure):
        x_grad = x.clone().requires_grad_(True)
        y_grad = y.clone().requires_grad_(True)
        grad_output = torch.randn(grad_shape, device=x.device, dtype=x.dtype)
        torch.cuda.synchronize()
        start.record()
        out = fn(x_grad, y_grad, sgp)
        out.backward(grad_output)
        end.record()
        torch.cuda.synchronize()
        total_time += start.elapsed_time(end)
    return total_time / n_measure


def run_correctness_tests(triton_fn, torch_fn, x, y, weight, sgp):
    """Run forward and backward correctness tests.
    """
    # Forward correctness check
    out_triton = triton_fn(x, y, weight)
    out_torch = torch_fn(x, y, sgp)

    print(
        f"Max absolute difference (fwd): {(out_torch - out_triton).abs().max().item():.1e}"
        + (" ✔" if torch.allclose(out_torch, out_triton, atol=1e-5) else " ✘")
    )

    # Backward correctness check
    x_torch = x.clone().detach().requires_grad_(True)
    y_torch = y.clone().detach().requires_grad_(True)

    x_triton = x.clone().detach().requires_grad_(True)
    y_triton = y.clone().detach().requires_grad_(True)
    weight_triton = weight.detach().clone().requires_grad_(True)

    out_torch = torch_fn(x_torch, y_torch, sgp)
    out_triton = triton_fn(x_triton, y_triton, weight_triton)

    grad_output = torch.randn_like(out_torch)

    out_torch.backward(grad_output)
    out_triton.backward(grad_output)

    print("\nBackward correctness:")
    print(
        f"grad_x max diff: {(x_torch.grad - x_triton.grad).abs().max().item():.1e}"
        + (" ✔" if torch.allclose(x_torch.grad, x_triton.grad, atol=1e-2) else " ✘")
    )
    print(
        f"grad_y max diff: {(y_torch.grad - y_triton.grad).abs().max().item():.1e}"
        + (" ✔" if torch.allclose(y_torch.grad, y_triton.grad, atol=1e-2) else " ✘")
    )

    return sgp, weight_triton


def run_benchmarks(triton_fn, torch_fn, x, y, weight, sgp, n_measure, warmup_iters=5):
    """Run forward and forward+backward benchmarks."""
    # Warm up
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = triton_fn(x, y, weight)
            _ = torch_fn(x, y, sgp)

    # Forward-only benchmark
    avg_time_fused = benchmark_forward(triton_fn, (x, y, weight), n_measure)
    avg_time_torch = benchmark_forward(torch_fn, (x, y, sgp), n_measure)

    print(f"\nFwd time (fused kernel): {avg_time_fused:.2f} ms")
    print(f"Fwd time (torch): {avg_time_torch:.2f} ms")
    print(f"Fwd Speedup: {avg_time_torch / avg_time_fused:.2f}x")

    # Forward + backward benchmark
    avg_time_fused_bwd = benchmark_backward_triton(triton_fn, x, y, weight, n_measure)
    avg_time_torch_bwd = benchmark_backward_torch(torch_fn, x, y, sgp, n_measure)

    print(f"\nFwd + bwd time (fused kernel): {avg_time_fused_bwd:.2f} ms")
    print(f"Fwd + bwd time (torch): {avg_time_torch_bwd:.2f} ms")
    print(f"Fwd + bwd Speedup: {avg_time_torch_bwd / avg_time_fused_bwd:.2f}x")
