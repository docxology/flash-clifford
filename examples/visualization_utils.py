#!/usr/bin/env python3
"""
Visualization utilities for Flash Clifford examples.

This module provides comprehensive plotting and visualization functions
for multivector analysis, performance metrics, and mathematical properties.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os
from datetime import datetime
import json


# Set style for consistent, publication-quality plots
plt.style.use('default')
sns.set_palette("husl")


def setup_plot_style():
    """Configure matplotlib for high-quality plots."""
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 2,
        'grid.alpha': 0.3,
    })


def plot_multivector_components(tensor, title="Multivector Components", save_path=None, dims=2):
    """Plot component-wise analysis of multivector tensors."""
    setup_plot_style()
    
    # Determine component names based on dimensions
    if dims == 2:
        component_names = ['Scalar', 'Vector X', 'Vector Y', 'Pseudoscalar']
    elif dims == 3:
        component_names = ['Scalar', 'Vector X', 'Vector Y', 'Vector Z', 
                          'Bivector XY', 'Bivector XZ', 'Bivector YZ', 'Pseudoscalar']
    else:
        component_names = [f'Component {i}' for i in range(tensor.shape[0])]
    
    num_components = tensor.shape[0]
    
    # Create subplots
    fig, axes = plt.subplots(2, (num_components + 1) // 2, figsize=(15, 8))
    if num_components <= 2:
        axes = axes.reshape(-1)
    else:
        axes = axes.flatten()
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for i in range(num_components):
        component_data = tensor[i].detach().cpu().numpy()
        
        # Plot histogram of component values
        axes[i].hist(component_data.flatten(), bins=50, alpha=0.7, 
                    color=sns.color_palette()[i % len(sns.color_palette())])
        axes[i].set_title(f'{component_names[i]}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
        
        # Add statistics text
        mean_val = np.mean(component_data)
        std_val = np.std(component_data)
        axes[i].text(0.02, 0.98, f'μ={mean_val:.3f}\nσ={std_val:.3f}', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(num_components, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        return fig


def plot_operation_flow(operations_data, title="Operation Flow Analysis", save_path=None):
    """Visualize the flow of operations and their effects."""
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Input/Output Magnitude Comparison
    ax1 = axes[0, 0]
    operations = list(operations_data.keys())
    input_norms = [operations_data[op]['input_norm'] for op in operations if 'input_norm' in operations_data[op]]
    output_norms = [operations_data[op]['output_norm'] for op in operations if 'output_norm' in operations_data[op]]
    
    if input_norms and output_norms:
        x = np.arange(len(operations))
        width = 0.35
        
        ax1.bar(x - width/2, input_norms, width, label='Input Norm', alpha=0.8)
        ax1.bar(x + width/2, output_norms, width, label='Output Norm', alpha=0.8)
        ax1.set_xlabel('Operations')
        ax1.set_ylabel('L2 Norm')
        ax1.set_title('Input vs Output Magnitudes')
        ax1.set_xticks(x)
        ax1.set_xticklabels(operations, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: Execution Times
    ax2 = axes[0, 1]
    exec_times = [operations_data[op]['exec_time'] for op in operations if 'exec_time' in operations_data[op]]
    
    if exec_times:
        bars = ax2.bar(operations, exec_times, alpha=0.8, color=sns.color_palette()[2])
        ax2.set_xlabel('Operations')
        ax2.set_ylabel('Execution Time (ms)')
        ax2.set_title('Operation Execution Times')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, time in zip(bars, exec_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.2f}ms', ha='center', va='bottom')
    
    # Plot 3: Memory Usage
    ax3 = axes[1, 0]
    memory_usage = [operations_data[op]['memory_mb'] for op in operations if 'memory_mb' in operations_data[op]]
    
    if memory_usage and any(mem > 0 for mem in memory_usage):
        # Filter out zero memory usage for pie chart
        non_zero_memory = [(op, mem) for op, mem in zip(operations, memory_usage) if mem > 0]
        if non_zero_memory:
            ops, mems = zip(*non_zero_memory)
            ax3.pie(mems, labels=ops, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Memory Usage Distribution')
        else:
            ax3.text(0.5, 0.5, 'No memory usage data\n(CPU mode)', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Memory Usage Distribution')
    else:
        ax3.text(0.5, 0.5, 'No memory usage data\n(CPU mode)', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Memory Usage Distribution')
    
    # Plot 4: Error Analysis
    ax4 = axes[1, 1]
    errors = [operations_data[op]['numerical_error'] for op in operations if 'numerical_error' in operations_data[op]]
    
    if errors:
        ax4.semilogy(operations, errors, 'o-', linewidth=2, markersize=8)
        ax4.set_xlabel('Operations')
        ax4.set_ylabel('Numerical Error (log scale)')
        ax4.set_title('Numerical Precision Analysis')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        return fig


def plot_performance_metrics(timing_data, title="Performance Analysis", save_path=None):
    """Plot performance timing and memory usage metrics."""
    setup_plot_style()
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Extract data
    operations = list(timing_data.keys())
    forward_times = [timing_data[op].get('forward_time', 0) for op in operations]
    backward_times = [timing_data[op].get('backward_time', 0) for op in operations]
    memory_usage = [timing_data[op].get('memory_mb', 0) for op in operations]
    
    # Plot 1: Forward vs Backward Times
    ax1 = fig.add_subplot(gs[0, :])
    x = np.arange(len(operations))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, forward_times, width, label='Forward Pass', alpha=0.8)
    bars2 = ax1.bar(x + width/2, backward_times, width, label='Backward Pass', alpha=0.8)
    
    ax1.set_xlabel('Operations')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Forward vs Backward Pass Timing')
    ax1.set_xticks(x)
    ax1.set_xticklabels(operations, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Memory Usage
    ax2 = fig.add_subplot(gs[1, 0])
    if any(mem > 0 for mem in memory_usage):
        bars = ax2.bar(operations, memory_usage, alpha=0.8, color=sns.color_palette()[2])
        ax2.set_xlabel('Operations')
        ax2.set_ylabel('Memory (MB)')
        ax2.set_title('Memory Usage by Operation')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        for bar, mem in zip(bars, memory_usage):
            if mem > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                        f'{mem:.1f}MB', ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No memory usage data\n(CPU mode)', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_xlabel('Operations')
        ax2.set_ylabel('Memory (MB)')
        ax2.set_title('Memory Usage by Operation')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Performance Efficiency (Operations per second)
    ax3 = fig.add_subplot(gs[1, 1])
    total_times = [f + b for f, b in zip(forward_times, backward_times)]
    efficiency = [1000/t if t > 0 else 0 for t in total_times]  # ops per second
    
    if any(eff > 0 for eff in efficiency):
        bars = ax3.bar(operations, efficiency, alpha=0.8, color=sns.color_palette()[3])
        ax3.set_xlabel('Operations')
        ax3.set_ylabel('Operations/Second')
        ax3.set_title('Performance Efficiency')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Scaling Analysis
    ax4 = fig.add_subplot(gs[2, :])
    if 'scaling_data' in timing_data:
        scaling = timing_data['scaling_data']
        sizes = scaling.get('sizes', [])
        times = scaling.get('times', [])
        
        if sizes and times:
            ax4.loglog(sizes, times, 'o-', linewidth=2, markersize=8, label='Measured')
            
            # Fit theoretical scaling curves
            if len(sizes) > 1:
                # Linear fit in log space
                log_sizes = np.log(sizes)
                log_times = np.log(times)
                coeffs = np.polyfit(log_sizes, log_times, 1)
                
                # Generate theoretical curve
                size_range = np.logspace(np.log10(min(sizes)), np.log10(max(sizes)), 100)
                theoretical = np.exp(coeffs[1]) * (size_range ** coeffs[0])
                
                ax4.loglog(size_range, theoretical, '--', alpha=0.7, 
                          label=f'Theoretical O(n^{coeffs[0]:.2f})')
            
            ax4.set_xlabel('Problem Size')
            ax4.set_ylabel('Time (ms)')
            ax4.set_title('Scaling Analysis')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        return fig


def plot_training_curves(training_history, title="Training Analysis", save_path=None):
    """Plot training curves and learning dynamics."""
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    epochs = training_history.get('epochs', [])
    losses = training_history.get('losses', [])
    accuracies = training_history.get('accuracies', [])
    
    # Plot 1: Loss Curve
    ax1 = axes[0, 0]
    if epochs and losses:
        ax1.plot(epochs, losses, 'o-', linewidth=2, markersize=6, color=sns.color_palette()[0])
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        if len(losses) > 2:
            z = np.polyfit(epochs, losses, 1)
            p = np.poly1d(z)
            ax1.plot(epochs, p(epochs), '--', alpha=0.7, color='red', 
                    label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
            ax1.legend()
    
    # Plot 2: Accuracy Curve
    ax2 = axes[0, 1]
    if epochs and accuracies:
        ax2.plot(epochs, accuracies, 'o-', linewidth=2, markersize=6, color=sns.color_palette()[1])
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1])
    
    # Plot 3: Loss Distribution
    ax3 = axes[1, 0]
    if losses:
        ax3.hist(losses, bins=min(20, len(losses)), alpha=0.7, color=sns.color_palette()[2])
        ax3.set_xlabel('Loss Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Loss Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Add statistics
        mean_loss = np.mean(losses)
        std_loss = np.std(losses)
        ax3.axvline(mean_loss, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_loss:.4f}')
        ax3.legend()
    
    # Plot 4: Learning Rate Analysis (if available)
    ax4 = axes[1, 1]
    learning_rates = training_history.get('learning_rates', [])
    if epochs and learning_rates:
        ax4.semilogy(epochs, learning_rates, 'o-', linewidth=2, markersize=6, color=sns.color_palette()[3])
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate (log scale)')
        ax4.set_title('Learning Rate Schedule')
        ax4.grid(True, alpha=0.3)
    else:
        # Plot loss vs accuracy correlation
        if losses and accuracies and len(losses) == len(accuracies):
            ax4.scatter(losses, accuracies, alpha=0.7, color=sns.color_palette()[3])
            ax4.set_xlabel('Loss')
            ax4.set_ylabel('Accuracy')
            ax4.set_title('Loss vs Accuracy Correlation')
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        return fig


def plot_mathematical_properties(property_data, title="Mathematical Properties", save_path=None):
    """Plot mathematical property verification results."""
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Equivariance Error Analysis
    ax1 = axes[0, 0]
    if 'equivariance_errors' in property_data:
        errors = property_data['equivariance_errors']
        operations = list(errors.keys())
        error_values = list(errors.values())
        
        bars = ax1.bar(operations, error_values, alpha=0.8, color=sns.color_palette()[0])
        ax1.set_xlabel('Operations')
        ax1.set_ylabel('Equivariance Error')
        ax1.set_title('Equivariance Verification')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Add threshold line
        threshold = 1e-3
        ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                   label=f'Threshold: {threshold}')
        ax1.legend()
    
    # Plot 2: Numerical Stability
    ax2 = axes[0, 1]
    if 'stability_analysis' in property_data:
        stability = property_data['stability_analysis']
        scales = list(stability.keys())
        max_errors = [stability[scale].get('max_error', 0) for scale in scales]
        
        ax2.loglog([float(s.replace('scale_', '')) for s in scales], max_errors, 
                  'o-', linewidth=2, markersize=8, color=sns.color_palette()[1])
        ax2.set_xlabel('Input Scale')
        ax2.set_ylabel('Maximum Error')
        ax2.set_title('Numerical Stability Analysis')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Grade Preservation
    ax3 = axes[1, 0]
    if 'grade_preservation' in property_data:
        grades = property_data['grade_preservation']
        grade_names = list(grades.keys())
        preservation_scores = [grades[grade].get('preservation_score', 0) for grade in grade_names]
        
        bars = ax3.bar(grade_names, preservation_scores, alpha=0.8, color=sns.color_palette()[2])
        ax3.set_xlabel('Multivector Grades')
        ax3.set_ylabel('Preservation Score')
        ax3.set_title('Grade Preservation Analysis')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim([0, 1])
        ax3.grid(True, alpha=0.3)
        
        # Add perfect preservation line
        ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Perfect')
        ax3.legend()
    
    # Plot 4: Operation Composition
    ax4 = axes[1, 1]
    if 'composition_analysis' in property_data:
        comp_data = property_data['composition_analysis']
        compositions = list(comp_data.keys())
        associativity_errors = [comp_data[comp].get('associativity_error', 0) for comp in compositions]
        
        ax4.semilogy(range(len(compositions)), associativity_errors, 
                    'o-', linewidth=2, markersize=8, color=sns.color_palette()[3])
        ax4.set_xlabel('Operation Compositions')
        ax4.set_ylabel('Associativity Error (log scale)')
        ax4.set_title('Operation Composition Analysis')
        ax4.set_xticks(range(len(compositions)))
        ax4.set_xticklabels(compositions, rotation=45)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        plt.show()
        return fig


def create_visualization_summary(output_dir, all_plots):
    """Create a summary of all generated visualizations."""
    viz_dir = os.path.join(output_dir, "visualizations")
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_plots": len(all_plots),
        "plots": {}
    }
    
    for plot_name, plot_path in all_plots.items():
        if os.path.exists(plot_path):
            file_size = os.path.getsize(plot_path)
            summary["plots"][plot_name] = {
                "path": plot_path,
                "size_kb": round(file_size / 1024, 2),
                "exists": True
            }
        else:
            summary["plots"][plot_name] = {
                "path": plot_path,
                "exists": False,
                "error": "File not found"
            }
    
    with open(os.path.join(viz_dir, "visualization_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary
