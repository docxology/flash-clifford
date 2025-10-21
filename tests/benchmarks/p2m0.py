import torch
torch.set_float32_matmul_precision('medium')
torch._dynamo.config.cache_size_limit = 512

from ops.p2m0 import fused_gelu_sgp_norm_2d
from tests.baselines import gelu_sgp_norm_2d_torch
from tests.utils import plot_heatmap, print_results_table, run_sweep


def setup_benchmark(batch_size, num_features):
    x = torch.randn(4, batch_size, num_features).cuda().contiguous()
    y = torch.randn(4, batch_size, num_features).cuda().contiguous()
    weight = torch.randn(num_features, 10).cuda().contiguous()
    return x, y, weight


if __name__ == "__main__":
    assert torch.cuda.is_available()

    path = "tests/benchmarks/results/p2m0"

    results = run_sweep(
        fused_gelu_sgp_norm_2d,
        gelu_sgp_norm_2d_torch,
        setup_benchmark,
        batch_sizes=[1024, 2048, 4096, 8192],
        num_features_list=[128, 256, 512, 1024],
        rep=200
    )

    print_results_table(results, "p2m0")

    plot_heatmap(results, 'speedup_fwd', 'Forward Pass Speedup: Triton vs PyTorch\nCl(2,0)',
                 path + '/speedup/fwd.png')
    plot_heatmap(results, 'speedup_fwd_bwd', 'Forward + Backward Pass Speedup: Triton vs PyTorch\nCl(2,0)',
                 path + '/speedup/fwd_bwd.png')
    plot_heatmap(results, 'mem_ratio_fwd', 'Forward Pass Memory Ratio: Fused / PyTorch\nCl(2,0)',
                 path + '/memory/fwd.png', invert_cmap=True)
    plot_heatmap(results, 'mem_ratio_fwd_bwd', 'Forward + Backward Pass Memory Ratio: Fused / PyTorch\nCl(2,0)',
                 path + '/memory/fwd_bwd.png', invert_cmap=True)
