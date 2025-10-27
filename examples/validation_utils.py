#!/usr/bin/env python3
"""
Validation utilities for Flash Clifford examples.

This module provides comprehensive validation functions for mathematical
properties, equivariance testing, and numerical stability analysis.
"""

import torch
import numpy as np
import json
import csv
import os
from datetime import datetime
import platform
import psutil


def export_tensor_data(tensor, filepath, name="tensor"):
    """Export tensor data to CSV with metadata."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Flatten tensor for CSV export
    original_shape = tensor.shape
    
    if tensor.dim() == 0:
        # Scalar tensor
        tensor_2d = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim() == 1:
        # 1D tensor - make it a column vector
        tensor_2d = tensor.unsqueeze(1)
    elif tensor.dim() == 2:
        # Already 2D
        tensor_2d = tensor
    else:
        # Higher dimensions - reshape to 2D
        tensor_2d = tensor.view(tensor.shape[0], -1)
    
    # Convert to numpy for CSV export
    data = tensor_2d.detach().cpu().numpy()
    
    # Write CSV with headers
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write metadata as comments
        writer.writerow([f"# {name} - Shape: {original_shape}"])
        writer.writerow([f"# Dtype: {tensor.dtype}"])
        writer.writerow([f"# Device: {tensor.device}"])
        writer.writerow([f"# Exported: {datetime.now().isoformat()}"])
        writer.writerow([])  # Empty row
        
        # Write column headers
        if data.shape[1] == 1:
            headers = ["value"]
        else:
            headers = [f"dim_{i}" for i in range(data.shape[1])]
            if tensor.dim() > 2:
                headers = [f"component_{i//np.prod(original_shape[1:])}_feature_{i%np.prod(original_shape[1:])}" 
                          for i in range(data.shape[1])]
        writer.writerow(headers)
        
        # Write data
        for row in data:
            if data.shape[1] == 1:
                writer.writerow([row.item() if hasattr(row, 'item') else row[0]])
            else:
                writer.writerow(row.tolist())


def export_gradients(tensors_with_grads, output_dir):
    """Export gradient information for tensors."""
    grad_dir = os.path.join(output_dir, "raw_data")
    os.makedirs(grad_dir, exist_ok=True)
    
    gradient_info = {}
    
    for name, tensor in tensors_with_grads.items():
        if tensor.grad is not None:
            # Export gradient tensor
            grad_path = os.path.join(grad_dir, f"grad_{name}.csv")
            export_tensor_data(tensor.grad, grad_path, f"grad_{name}")
            
            # Collect gradient statistics
            gradient_info[name] = {
                "has_gradient": True,
                "grad_shape": list(tensor.grad.shape),
                "grad_norm": tensor.grad.norm().item(),
                "grad_mean": tensor.grad.mean().item(),
                "grad_std": tensor.grad.std().item(),
                "grad_min": tensor.grad.min().item(),
                "grad_max": tensor.grad.max().item(),
            }
        else:
            gradient_info[name] = {"has_gradient": False}
    
    # Save gradient summary
    with open(os.path.join(grad_dir, "gradient_summary.json"), 'w') as f:
        json.dump(gradient_info, f, indent=2)
    
    return gradient_info


def export_system_info(output_dir):
    """Export hardware and software configuration."""
    info_dir = os.path.join(output_dir, "raw_data")
    os.makedirs(info_dir, exist_ok=True)
    
    system_info = {
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "hardware": {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
    }
    
    if torch.cuda.is_available():
        system_info["cuda"] = {
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
        }
    
    with open(os.path.join(info_dir, "system_info.json"), 'w') as f:
        json.dump(system_info, f, indent=2)
    
    return system_info


def validate_mathematical_properties(x, y, operations_dict, output_dir):
    """Verify Clifford algebra mathematical properties."""
    validation_results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }
    
    # Test 1: Linearity of operations
    if 'gelu' in operations_dict:
        gelu_fn = operations_dict['gelu']
        
        # Test additivity (approximately, since GELU is not linear)
        x1, x2 = x[:, :, :x.shape[2]//2], x[:, :, x.shape[2]//2:]
        alpha, beta = 0.5, 0.3
        
        combined_input = alpha * x1 + beta * x2
        gelu_combined = gelu_fn(combined_input)
        gelu_separate = alpha * gelu_fn(x1) + beta * gelu_fn(x2)
        
        linearity_error = (gelu_combined - gelu_separate).abs().max().item()
        validation_results["tests"]["gelu_approximate_linearity"] = {
            "max_error": linearity_error,
            "passed": linearity_error < 1.0,  # GELU is not linear, so we expect some error
            "description": "GELU approximate linearity test (expected to have some error)"
        }
    
    # Test 2: Shape preservation
    shape_tests = {}
    for name, op_fn in operations_dict.items():
        try:
            if name in ['sgp_2d', 'sgp_3d', 'fcgp_2d', 'fcgp_3d']:
                # These operations need weight parameter
                continue
            
            result = op_fn(x)
            shape_preserved = result.shape == x.shape
            shape_tests[name] = {
                "input_shape": list(x.shape),
                "output_shape": list(result.shape),
                "preserved": shape_preserved
            }
        except Exception as e:
            shape_tests[name] = {"error": str(e)}
    
    validation_results["tests"]["shape_preservation"] = shape_tests
    
    # Test 3: Finite output verification
    finite_tests = {}
    for name, op_fn in operations_dict.items():
        try:
            if name in ['sgp_2d', 'sgp_3d', 'fcgp_2d', 'fcgp_3d']:
                continue
                
            result = op_fn(x)
            all_finite = torch.isfinite(result).all().item()
            finite_tests[name] = {
                "all_finite": all_finite,
                "num_nan": torch.isnan(result).sum().item(),
                "num_inf": torch.isinf(result).sum().item(),
            }
        except Exception as e:
            finite_tests[name] = {"error": str(e)}
    
    validation_results["tests"]["finite_outputs"] = finite_tests
    
    # Save validation results
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    with open(os.path.join(reports_dir, "mathematical_validation.json"), 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    return validation_results


def test_equivariance(layer_fn, x, output_dir, dims=2):
    """Test equivariance properties with rotation matrices."""
    validation_results = {
        "timestamp": datetime.now().isoformat(),
        "dimensions": dims,
        "tests": {}
    }
    
    if dims == 2:
        # Create 2D rotation matrix
        angle = np.pi / 4  # 45 degrees
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Rotation matrix for 2D vectors
        R = torch.tensor([
            [1, 0, 0, 0],      # scalar unchanged
            [0, cos_a, -sin_a, 0],  # vector_x
            [0, sin_a, cos_a, 0],   # vector_y  
            [0, 0, 0, 1]       # pseudoscalar unchanged
        ], dtype=x.dtype, device=x.device)
        
    elif dims == 3:
        # Create 3D rotation matrix (rotation around z-axis)
        angle = np.pi / 6  # 30 degrees
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        
        # Rotation matrix for 3D vectors (simplified)
        R = torch.eye(8, dtype=x.dtype, device=x.device)
        R[1, 1] = cos_a   # vector_x
        R[1, 2] = -sin_a
        R[2, 1] = sin_a   # vector_y
        R[2, 2] = cos_a
    
    try:
        # Apply rotation to input
        x_rotated = torch.einsum('ij,jkl->ikl', R, x)
        
        # Compute outputs
        output_original = layer_fn(x)
        output_rotated = layer_fn(x_rotated)
        
        # Apply rotation to original output
        output_original_rotated = torch.einsum('ij,jkl->ikl', R, output_original)
        
        # Check equivariance: R(f(x)) ≈ f(R(x))
        equivariance_error = (output_original_rotated - output_rotated).abs().max().item()
        
        validation_results["tests"]["rotation_equivariance"] = {
            "rotation_angle_degrees": np.degrees(angle),
            "max_error": equivariance_error,
            "passed": equivariance_error < 1e-3,
            "description": f"Rotation equivariance test for {dims}D"
        }
        
    except Exception as e:
        validation_results["tests"]["rotation_equivariance"] = {
            "error": str(e),
            "passed": False
        }
    
    # Save equivariance results
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    with open(os.path.join(reports_dir, f"equivariance_test_{dims}d.json"), 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    return validation_results


def analyze_numerical_stability(operations_dict, output_dir, test_scales=[1e-6, 1e-3, 1.0, 1e3, 1e6]):
    """Analyze numerical stability across different input scales."""
    stability_results = {
        "timestamp": datetime.now().isoformat(),
        "test_scales": test_scales,
        "results": {}
    }
    
    for scale in test_scales:
        scale_results = {}
        
        # Create test input at this scale
        x_test = torch.randn(4, 8, 16) * scale
        
        for name, op_fn in operations_dict.items():
            try:
                if name in ['sgp_2d', 'sgp_3d', 'fcgp_2d', 'fcgp_3d']:
                    continue
                    
                result = op_fn(x_test)
                
                scale_results[name] = {
                    "input_scale": scale,
                    "output_mean": result.mean().item(),
                    "output_std": result.std().item(),
                    "output_min": result.min().item(),
                    "output_max": result.max().item(),
                    "all_finite": torch.isfinite(result).all().item(),
                    "dynamic_range": (result.max() - result.min()).item(),
                }
                
            except Exception as e:
                scale_results[name] = {"error": str(e)}
        
        stability_results["results"][f"scale_{scale:.0e}"] = scale_results
    
    # Save stability analysis
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    with open(os.path.join(reports_dir, "numerical_stability.json"), 'w') as f:
        json.dump(stability_results, f, indent=2)
    
    return stability_results


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def generate_validation_report(output_dir, validation_data):
    """Generate a comprehensive HTML validation report."""
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Convert numpy types for JSON serialization
    serializable_data = convert_numpy_types(validation_data)
    
    # Create a summary of key metrics
    summary_info = ""
    if 'model_summary' in serializable_data:
        model_summary = serializable_data['model_summary']
        summary_info = f"""
        <div class="section">
            <h2>📈 Model Summary</h2>
            <ul>
                <li><strong>Total Parameters:</strong> {model_summary.get('total_parameters', 'N/A'):,}</li>
                <li><strong>Trainable Parameters:</strong> {model_summary.get('trainable_parameters', 'N/A'):,}</li>
                <li><strong>Final Loss:</strong> {model_summary.get('final_loss', 'N/A'):.4f}</li>
                <li><strong>Final Accuracy:</strong> {model_summary.get('final_accuracy', 'N/A'):.4f}</li>
                <li><strong>Best Accuracy:</strong> {model_summary.get('best_accuracy', 'N/A'):.4f}</li>
                <li><strong>Convergence Epoch:</strong> {model_summary.get('convergence_epoch', 'N/A')}</li>
            </ul>
        </div>
        """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Flash Clifford Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; padding: 15px; border-left: 3px solid #007acc; }}
            .pass {{ color: green; font-weight: bold; }}
            .fail {{ color: red; font-weight: bold; }}
            .warning {{ color: orange; font-weight: bold; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .code {{ background: #f5f5f5; padding: 10px; font-family: monospace; white-space: pre-wrap; max-height: 400px; overflow-y: auto; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🧪 Flash Clifford Validation Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>System:</strong> {platform.system()} {platform.release()}</p>
            <p><strong>PyTorch:</strong> {torch.__version__}</p>
        </div>
        
        <div class="section">
            <h2>📊 Validation Summary</h2>
            <p>This report contains comprehensive validation results for Flash Clifford operations.</p>
        </div>
        
        {summary_info}
        
        <div class="section">
            <h2>🔍 Detailed Results</h2>
            <div class="code">
{json.dumps(serializable_data, indent=2)}
            </div>
        </div>
        
        <div class="section">
            <h2>✅ Recommendations</h2>
            <ul>
                <li>All operations should preserve tensor shapes</li>
                <li>Outputs should be finite (no NaN or Inf values)</li>
                <li>Equivariance properties should be maintained</li>
                <li>Numerical stability should be verified across scales</li>
                <li>Training curves should show convergence</li>
                <li>Model weights should be well-distributed</li>
            </ul>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(reports_dir, "validation_report.html"), 'w') as f:
        f.write(html_content)
    
    return os.path.join(reports_dir, "validation_report.html")
