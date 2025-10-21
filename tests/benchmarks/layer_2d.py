import torch
torch.set_float32_matmul_precision('medium')
torch._dynamo.config.cache_size_limit = 512

from modules.layer import Layer
from modules.baseline import Layer as BaselineLayer
from tests.utils import plot_heatmap, print_results_table, run_single_benchmark


def setup_benchmark(batch_size, num_features):
    x = torch.randn(4, batch_size, num_features).cuda().contiguous()
    return (x,)


def create_layers(num_features: int):
    layer_fused = Layer(num_features, dims=2, normalize=True, use_fc=False).float().cuda()
    layer_torch = BaselineLayer(num_features, dims=2, normalize=True, use_fc=False).float().cuda()
    return layer_fused, layer_torch


if __name__ == "__main__":
    assert torch.cuda.is_available()
    
    rep = 1000
    warmup = 500
    batch_sizes=[1024, 2048, 4096, 8192]
    num_features_list=[128, 256, 512, 1024]
    path = "tests/benchmarks/results/layer_2d"
    
    results = []

    print("Running benchmark sweep...")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Num features: {num_features_list}")

    for batch_size in batch_sizes:
        for num_features in num_features_list:
            print(f"Running batch_size={batch_size}, num_features={num_features}...", end=" ")
            triton_fn, torch_fn = create_layers(num_features)
            triton_fn = torch.compile(triton_fn)
            torch_fn = torch.compile(torch_fn)
            
            result = run_single_benchmark(
                triton_fn, torch_fn, setup_benchmark, batch_size,
                num_features, rep, warmup, verify_correctness=False, return_mode='mean'
            )
            results.append(result)

            fwd_msg = (f"Fwd: {result['speedup_fwd']:.2f}x" 
                      if result['speedup_fwd'] else "Fwd: OOM")
            bwd_msg = (f"Fwd+Bwd: {result['speedup_fwd_bwd']:.2f}x" 
                      if result['speedup_fwd_bwd'] else "Fwd+Bwd: OOM")
            print(f"{fwd_msg}, {bwd_msg}")
    
    print_results_table(results, "layer_2d")

    plot_heatmap(results, 'speedup_fwd', 'Forward Pass Speedup: Triton vs PyTorch\nCl(2,0)',
                 path + '/speedup/fwd.png')
    plot_heatmap(results, 'speedup_fwd_bwd', 'Forward + Backward Pass Speedup: Triton vs PyTorch\nCl(2,0)',
                 path + '/speedup/fwd_bwd.png')
    plot_heatmap(results, 'mem_ratio_fwd', 'Forward Pass Memory Ratio: Fused / PyTorch\nCl(2,0)',
                 path + '/memory/fwd.png', invert_cmap=True)
    plot_heatmap(results, 'mem_ratio_fwd_bwd', 'Forward + Backward Pass Memory Ratio: Fused / PyTorch\nCl(2,0)',
                 path + '/memory/fwd_bwd.png', invert_cmap=True)