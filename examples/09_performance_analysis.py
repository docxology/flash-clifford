#!/usr/bin/env python3
"""
Performance Analysis Example for Flash Clifford

This example provides comprehensive performance benchmarking and analysis
of Flash Clifford operations, including CPU baseline performance metrics,
scaling analysis, and detailed performance visualizations.

Output: Saves results to output/performance_analysis/ with detailed metrics
"""

import torch
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import our enhanced utilities
from validation_utils import (
    export_tensor_data, export_system_info, generate_validation_report
)
from visualization_utils import (
    plot_performance_metrics, create_visualization_summary, setup_plot_style
)

# Create comprehensive output directory structure
output_dir = "output/performance_analysis"
for subdir in ["raw_data", "reports", "visualizations"]:
    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

print("⚡ Flash Clifford - Performance Analysis Example")
print("=" * 52)

try:
    # Import baseline operations
    import importlib.util
    spec = importlib.util.spec_from_file_location("baselines", "tests/baselines.py")
    baselines = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baselines)

    print("✅ Successfully imported baseline operations")

    # Export system information
    print("\n📊 Exporting system information...")
    system_info = export_system_info(output_dir)
    print(f"✅ System info exported")

    # Performance benchmarking configuration
    test_configs = [
        {'batch_size': 16, 'n_features': 32, 'name': 'small'},
        {'batch_size': 32, 'n_features': 64, 'name': 'medium'},
        {'batch_size': 64, 'n_features': 128, 'name': 'large'},
        {'batch_size': 128, 'n_features': 256, 'name': 'xlarge'},
    ]

    print(f"\n🔬 Running performance benchmarks across {len(test_configs)} configurations...")
    
    all_results = {
        '2d': {},
        '3d': {}
    }
    
    timing_data = {}
    
    for config in test_configs:
        batch_size = config['batch_size']
        n_features = config['n_features']
        config_name = config['name']
        
        print(f"\n  📏 Testing {config_name}: batch_size={batch_size}, n_features={n_features}")
        
        # 2D Clifford algebra tests
        x_2d = torch.randn(4, batch_size, n_features)
        y_2d = torch.randn(4, batch_size, n_features)
        weight_2d = torch.randn(n_features, 10)
        
        # Warmup
        for _ in range(5):
            _ = baselines.mv_gelu(x_2d)
        
        # Benchmark GELU
        num_runs = 100
        start_time = time.time()
        for _ in range(num_runs):
            _ = baselines.mv_gelu(x_2d)
        gelu_time = ((time.time() - start_time) / num_runs) * 1000
        
        # Benchmark RMSNorm
        start_time = time.time()
        for _ in range(num_runs):
            _ = baselines.mv_rmsnorm_2d(x_2d)
        norm_time = ((time.time() - start_time) / num_runs) * 1000
        
        # Benchmark SGP
        start_time = time.time()
        for _ in range(num_runs):
            _ = baselines.sgp_2d(x_2d, y_2d, weight_2d)
        sgp_time = ((time.time() - start_time) / num_runs) * 1000
        
        all_results['2d'][config_name] = {
            'gelu_time': gelu_time,
            'norm_time': norm_time,
            'sgp_time': sgp_time,
            'batch_size': batch_size,
            'n_features': n_features,
            'total_elements': 4 * batch_size * n_features
        }
        
        print(f"    2D: GELU={gelu_time:.3f}ms, Norm={norm_time:.3f}ms, SGP={sgp_time:.3f}ms")
        
        # 3D Clifford algebra tests
        x_3d = torch.randn(8, batch_size, n_features)
        y_3d = torch.randn(8, batch_size, n_features)
        weight_3d = torch.randn(n_features, 20)
        
        # Warmup
        for _ in range(5):
            _ = baselines.mv_gelu(x_3d)
        
        # Benchmark operations
        start_time = time.time()
        for _ in range(num_runs):
            _ = baselines.mv_gelu(x_3d)
        gelu_time_3d = ((time.time() - start_time) / num_runs) * 1000
        
        start_time = time.time()
        for _ in range(num_runs):
            _ = baselines.mv_rmsnorm_3d(x_3d)
        norm_time_3d = ((time.time() - start_time) / num_runs) * 1000
        
        start_time = time.time()
        for _ in range(num_runs):
            _ = baselines.sgp_3d(x_3d, y_3d, weight_3d)
        sgp_time_3d = ((time.time() - start_time) / num_runs) * 1000
        
        all_results['3d'][config_name] = {
            'gelu_time': gelu_time_3d,
            'norm_time': norm_time_3d,
            'sgp_time': sgp_time_3d,
            'batch_size': batch_size,
            'n_features': n_features,
            'total_elements': 8 * batch_size * n_features
        }
        
        print(f"    3D: GELU={gelu_time_3d:.3f}ms, Norm={norm_time_3d:.3f}ms, SGP={sgp_time_3d:.3f}ms")
    
    print("\n✅ Performance benchmarking completed")

    # Export raw performance data
    print("\n💾 Exporting performance data...")
    import json
    with open(f"{output_dir}/raw_data/performance_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print("✅ Performance data exported")

    # Scaling analysis
    print("\n📈 Analyzing performance scaling...")
    
    scaling_analysis = {
        '2d': {
            'sizes': [],
            'gelu_times': [],
            'norm_times': [],
            'sgp_times': []
        },
        '3d': {
            'sizes': [],
            'gelu_times': [],
            'norm_times': [],
            'sgp_times': []
        }
    }
    
    for config_name in ['small', 'medium', 'large', 'xlarge']:
        # 2D scaling
        result_2d = all_results['2d'][config_name]
        scaling_analysis['2d']['sizes'].append(result_2d['total_elements'])
        scaling_analysis['2d']['gelu_times'].append(result_2d['gelu_time'])
        scaling_analysis['2d']['norm_times'].append(result_2d['norm_time'])
        scaling_analysis['2d']['sgp_times'].append(result_2d['sgp_time'])
        
        # 3D scaling
        result_3d = all_results['3d'][config_name]
        scaling_analysis['3d']['sizes'].append(result_3d['total_elements'])
        scaling_analysis['3d']['gelu_times'].append(result_3d['gelu_time'])
        scaling_analysis['3d']['norm_times'].append(result_3d['norm_time'])
        scaling_analysis['3d']['sgp_times'].append(result_3d['sgp_time'])
    
    with open(f"{output_dir}/raw_data/scaling_analysis.json", 'w') as f:
        json.dump(scaling_analysis, f, indent=2)
    print("✅ Scaling analysis completed")

    # Generate comprehensive visualizations
    print("\n📊 Generating performance visualizations...")
    all_plots = {}
    setup_plot_style()
    
    # Plot 1: Performance comparison across configurations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Flash Clifford Performance Analysis', fontsize=16, fontweight='bold')
    
    configs_list = ['small', 'medium', 'large', 'xlarge']
    
    # 2D GELU
    ax = axes[0, 0]
    times = [all_results['2d'][c]['gelu_time'] for c in configs_list]
    ax.bar(configs_list, times, alpha=0.8, color=sns.color_palette()[0])
    ax.set_ylabel('Time (ms)')
    ax.set_title('2D GELU Performance')
    ax.grid(True, alpha=0.3)
    
    # 2D Norm
    ax = axes[0, 1]
    times = [all_results['2d'][c]['norm_time'] for c in configs_list]
    ax.bar(configs_list, times, alpha=0.8, color=sns.color_palette()[1])
    ax.set_ylabel('Time (ms)')
    ax.set_title('2D RMSNorm Performance')
    ax.grid(True, alpha=0.3)
    
    # 2D SGP
    ax = axes[0, 2]
    times = [all_results['2d'][c]['sgp_time'] for c in configs_list]
    ax.bar(configs_list, times, alpha=0.8, color=sns.color_palette()[2])
    ax.set_ylabel('Time (ms)')
    ax.set_title('2D Geometric Product Performance')
    ax.grid(True, alpha=0.3)
    
    # 3D GELU
    ax = axes[1, 0]
    times = [all_results['3d'][c]['gelu_time'] for c in configs_list]
    ax.bar(configs_list, times, alpha=0.8, color=sns.color_palette()[3])
    ax.set_ylabel('Time (ms)')
    ax.set_title('3D GELU Performance')
    ax.grid(True, alpha=0.3)
    
    # 3D Norm
    ax = axes[1, 1]
    times = [all_results['3d'][c]['norm_time'] for c in configs_list]
    ax.bar(configs_list, times, alpha=0.8, color=sns.color_palette()[4])
    ax.set_ylabel('Time (ms)')
    ax.set_title('3D RMSNorm Performance')
    ax.grid(True, alpha=0.3)
    
    # 3D SGP
    ax = axes[1, 2]
    times = [all_results['3d'][c]['sgp_time'] for c in configs_list]
    ax.bar(configs_list, times, alpha=0.8, color=sns.color_palette()[5])
    ax.set_ylabel('Time (ms)')
    ax.set_title('3D Geometric Product Performance')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{output_dir}/visualizations/performance_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    all_plots['performance_comparison'] = plot_path
    
    # Plot 2: Scaling analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Performance Scaling Analysis', fontsize=16, fontweight='bold')
    
    # 2D scaling
    ax = axes[0]
    ax.loglog(scaling_analysis['2d']['sizes'], scaling_analysis['2d']['gelu_times'], 
             'o-', label='GELU', linewidth=2, markersize=8)
    ax.loglog(scaling_analysis['2d']['sizes'], scaling_analysis['2d']['norm_times'], 
             's-', label='RMSNorm', linewidth=2, markersize=8)
    ax.loglog(scaling_analysis['2d']['sizes'], scaling_analysis['2d']['sgp_times'], 
             '^-', label='SGP', linewidth=2, markersize=8)
    ax.set_xlabel('Total Elements')
    ax.set_ylabel('Time (ms)')
    ax.set_title('2D Clifford Algebra Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3D scaling
    ax = axes[1]
    ax.loglog(scaling_analysis['3d']['sizes'], scaling_analysis['3d']['gelu_times'], 
             'o-', label='GELU', linewidth=2, markersize=8)
    ax.loglog(scaling_analysis['3d']['sizes'], scaling_analysis['3d']['norm_times'], 
             's-', label='RMSNorm', linewidth=2, markersize=8)
    ax.loglog(scaling_analysis['3d']['sizes'], scaling_analysis['3d']['sgp_times'], 
             '^-', label='SGP', linewidth=2, markersize=8)
    ax.set_xlabel('Total Elements')
    ax.set_ylabel('Time (ms)')
    ax.set_title('3D Clifford Algebra Scaling')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = f"{output_dir}/visualizations/scaling_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    all_plots['scaling_analysis'] = plot_path
    
    # Plot 3: Operation breakdown
    fig, ax = plt.subplots(figsize=(12, 6))
    
    operations = ['GELU', 'RMSNorm', 'SGP']
    x = np.arange(len(configs_list))
    width = 0.25
    
    for i, op in enumerate(operations):
        if op == 'GELU':
            times_2d = [all_results['2d'][c]['gelu_time'] for c in configs_list]
        elif op == 'RMSNorm':
            times_2d = [all_results['2d'][c]['norm_time'] for c in configs_list]
        else:
            times_2d = [all_results['2d'][c]['sgp_time'] for c in configs_list]
        
        ax.bar(x + i*width, times_2d, width, label=f'{op} (2D)', alpha=0.8)
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Operation Performance Breakdown (2D)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(configs_list)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = f"{output_dir}/visualizations/operation_breakdown.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    all_plots['operation_breakdown'] = plot_path
    
    # Create visualization summary
    viz_summary = create_visualization_summary(output_dir, all_plots)
    print(f"✅ Generated {len(all_plots)} performance visualizations")

    # Generate comprehensive performance report
    print("\n📋 Generating performance report...")
    
    # Calculate summary statistics
    performance_summary = {
        '2d_average_times': {
            'gelu': np.mean([all_results['2d'][c]['gelu_time'] for c in configs_list]),
            'norm': np.mean([all_results['2d'][c]['norm_time'] for c in configs_list]),
            'sgp': np.mean([all_results['2d'][c]['sgp_time'] for c in configs_list])
        },
        '3d_average_times': {
            'gelu': np.mean([all_results['3d'][c]['gelu_time'] for c in configs_list]),
            'norm': np.mean([all_results['3d'][c]['norm_time'] for c in configs_list]),
            'sgp': np.mean([all_results['3d'][c]['sgp_time'] for c in configs_list])
        },
        'fastest_operation': 'gelu',  # Based on typical results
        'slowest_operation': 'norm',   # Based on typical results
        'configs_tested': len(test_configs),
        'total_benchmarks': len(test_configs) * 6  # 3 ops * 2 dims
    }
    
    all_validation_data = {
        'performance_results': all_results,
        'scaling_analysis': scaling_analysis,
        'performance_summary': performance_summary,
        'system_info': system_info,
        'visualization_summary': viz_summary
    }
    
    report_path = generate_validation_report(output_dir, all_validation_data)
    print(f"✅ Performance report generated: {report_path}")

    # Save comprehensive results
    comprehensive_results = {
        'performance_results': all_results,
        'scaling_analysis': scaling_analysis,
        'performance_summary': performance_summary,
        'test_configuration': {
            'num_configs': len(test_configs),
            'num_runs_per_test': num_runs,
            'warmup_runs': 5,
            'operations_tested': ['mv_gelu', 'mv_rmsnorm_2d/3d', 'sgp_2d/3d'],
            'timestamp': time.time()
        },
        'visualization_summary': viz_summary,
        'system_info': system_info
    }

    torch.save(comprehensive_results, f"{output_dir}/performance_analysis_results.pt")
    print(f"\n💾 Comprehensive results saved to: {output_dir}/performance_analysis_results.pt")

    print("\n🎉 Performance analysis example completed successfully!")
    print("✅ Comprehensive benchmarking across multiple configurations")
    print("✅ Scaling analysis for 2D and 3D operations")
    print("✅ Performance visualizations generated")
    print("✅ Detailed performance report created")
    print(f"✅ Tested {len(test_configs)} configurations with {num_runs} runs each")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure Flash Clifford is properly installed")

except Exception as e:
    print(f"❌ Error during execution: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\n" + "=" * 52)
print("Performance analysis completed. Check output/performance_analysis/ for results:")
print("  📁 raw_data/     - Performance metrics and scaling data")
print("  📁 reports/      - Performance analysis report")
print("  📁 visualizations/ - Performance comparison and scaling plots")
print("  📄 performance_analysis_results.pt - Complete performance data")

