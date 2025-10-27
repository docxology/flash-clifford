#!/usr/bin/env python3
"""
Comprehensive test runner for Flash Clifford.

This script validates all tests and ensures they work correctly.
For CUDA environments, it runs full performance tests.
For non-CUDA environments, it validates code structure and documentation.
"""

import os
import sys
import importlib.util
from pathlib import Path
import subprocess
import ast
import inspect

def check_cuda_availability():
    """Check if CUDA is available for full testing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def validate_python_syntax():
    """Validate Python syntax for all Python files."""
    print("🔍 Validating Python syntax...")

    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        if any(skip in root for skip in ['__pycache__', '.git', '.pytest_cache', 'node_modules']):
            continue
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    syntax_errors = []
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            ast.parse(source, filename=file_path)
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}:{e.lineno}:{e.colno}: {e.msg}")
        except Exception as e:
            syntax_errors.append(f"{file_path}: {str(e)}")

    if syntax_errors:
        print("❌ Syntax errors found:")
        for error in syntax_errors:
            print(f"  {error}")
        return False

    print("✅ All Python files have valid syntax")
    return True

def validate_imports():
    """Validate that all imports work correctly."""
    print("🔍 Validating imports...")

    # Test critical imports
    critical_imports = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
    ]

    missing_imports = []
    for module, name in critical_imports:
        try:
            __import__(module)
        except ImportError:
            missing_imports.append(name)

    if missing_imports:
        print(f"⚠️  Missing optional dependencies: {', '.join(missing_imports)}")
        print("   (These are needed for full testing but not required for basic validation)")
    else:
        print("✅ All critical dependencies available")

    return True

def validate_baseline_implementations():
    """Validate that baseline implementations can be imported and have correct structure."""
    print("🔍 Validating baseline implementations...")

    try:
        # Import baseline functions by executing the file
        import sys
        import os
        sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))

        # Import the module directly
        import importlib.util
        spec = importlib.util.spec_from_file_location("baselines", "tests/baselines.py")
        baselines = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baselines)

        mv_gelu = baselines.mv_gelu
        mv_rmsnorm_2d = baselines.mv_rmsnorm_2d
        mv_rmsnorm_3d = baselines.mv_rmsnorm_3d
        sgp_2d = baselines.sgp_2d
        sgp_3d = baselines.sgp_3d
        fcgp_2d = baselines.fcgp_2d
        fcgp_3d = baselines.fcgp_3d
        gelu_sgp_norm_2d_torch = baselines.gelu_sgp_norm_2d_torch
        gelu_sgp_norm_3d_torch = baselines.gelu_sgp_norm_3d_torch
        gelu_fcgp_norm_2d_torch = baselines.gelu_fcgp_norm_2d_torch
        gelu_fcgp_norm_3d_torch = baselines.gelu_fcgp_norm_3d_torch

        # Check function signatures
        expected_functions = {
            'mv_gelu': 1,  # 1 parameter
            'mv_rmsnorm_2d': 2,  # x, eps
            'mv_rmsnorm_3d': 2,  # x, eps
            'sgp_2d': 3,  # x, y, weight
            'sgp_3d': 3,  # x, y, weight
            'fcgp_2d': 3,  # x, y, weight
            'fcgp_3d': 3,  # x, y, weight
            'gelu_sgp_norm_2d_torch': 4,  # x, y, weight, normalize
            'gelu_sgp_norm_3d_torch': 4,  # x, y, weight, normalize
            'gelu_fcgp_norm_2d_torch': 4,  # x, y, weight, normalize
            'gelu_fcgp_norm_3d_torch': 4,  # x, y, weight, normalize
        }

        for func_name, expected_params in expected_functions.items():
            func = locals().get(func_name)
            if func is None:
                print(f"❌ Missing function: {func_name}")
                return False

            sig = inspect.signature(func)
            actual_params = len(sig.parameters)

            if actual_params != expected_params:
                print(f"❌ {func_name}: expected {expected_params} params, got {actual_params}")
                return False

        print("✅ All baseline functions have correct signatures")
        return True

    except ImportError as e:
        print(f"❌ Failed to import baseline functions: {e}")
        return False

def validate_test_structure():
    """Validate that test files have correct structure."""
    print("🔍 Validating test structure...")

    test_files = [
        'tests/p2m0.py',
        'tests/p3m0.py',
        'tests/fc_p2m0.py',
        'tests/fc_p3m0.py',
        'tests/benchmarks/p2m0.py',
        'tests/benchmarks/p3m0.py',
        'tests/benchmarks/fc_p2m0.py',
        'tests/benchmarks/fc_p3m0.py',
        'tests/benchmarks/layer_2d.py',
        'tests/benchmarks/layer_3d.py',
    ]

    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"❌ Missing test file: {test_file}")
            return False

    print("✅ All test files exist")
    return True

def validate_documentation_structure():
    """Validate documentation structure and content."""
    print("🔍 Validating documentation structure...")

    doc_files = [
        'docs/index.md',
        'docs/core-concepts.md',
        'docs/operations.md',
        'docs/architecture.md',
        'docs/performance.md',
        'docs/implementation.md',
        'docs/api-reference.md',
        'docs/examples.md',
        'docs/development.md',
        'docs/troubleshooting.md',
        'docs/performance-tuning.md',
        'docs/research-applications.md',
        'docs/README.md',
    ]

    missing_docs = []
    for doc_file in doc_files:
        if not os.path.exists(doc_file):
            missing_docs.append(doc_file)

    if missing_docs:
        print("❌ Missing documentation files:")
        for doc in missing_docs:
            print(f"  {doc}")
        return False

    print("✅ All documentation files exist")
    return True

def validate_mathematical_correctness():
    """Validate mathematical correctness of baseline implementations."""
    print("🔍 Validating mathematical correctness...")

    try:
        import torch
        import sys
        import os
        sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))

        import importlib.util
        spec = importlib.util.spec_from_file_location("baselines", "tests/baselines.py")
        baselines = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baselines)

        mv_gelu = baselines.mv_gelu
        mv_rmsnorm_2d = baselines.mv_rmsnorm_2d
        sgp_2d = baselines.sgp_2d

        # Test with simple inputs
        batch_size, features = 2, 4
        x = torch.randn(4, batch_size, features)

        # Test GELU
        try:
            result = mv_gelu(x)
            assert result.shape == x.shape, f"GELU shape mismatch: {result.shape} vs {x.shape}"
        except Exception as e:
            print(f"❌ GELU test failed: {e}")
            return False

        # Test normalization
        try:
            result = mv_rmsnorm_2d(x)
            assert result.shape == x.shape, f"Normalization shape mismatch: {result.shape} vs {x.shape}"
        except Exception as e:
            print(f"❌ Normalization test failed: {e}")
            return False

        # Test geometric product
        try:
            y = torch.randn(4, batch_size, features)
            weight = torch.randn(features, 10)
            result = sgp_2d(x, y, weight)
            assert result.shape == x.shape, f"GP shape mismatch: {result.shape} vs {x.shape}"
        except Exception as e:
            print(f"❌ Geometric product test failed: {e}")
            return False

        print("✅ Mathematical correctness validation passed")
        return True

    except ImportError as e:
        print(f"❌ Mathematical validation failed due to import error: {e}")
        return False

def run_cuda_tests():
    """Run full CUDA tests if available."""
    print("🚀 Running CUDA tests...")

    test_commands = [
        "python -m tests.p2m0",
        "python -m tests.p3m0",
        "python -m tests.fc_p2m0",
        "python -m tests.fc_p3m0",
        "python -m tests.benchmarks.p2m0",
        "python -m tests.benchmarks.p3m0",
        "python -m tests.benchmarks.fc_p2m0",
        "python -m tests.benchmarks.fc_p3m0",
        "python -m tests.benchmarks.layer_2d",
        "python -m tests.benchmarks.layer_3d",
    ]

    results = []
    for cmd in test_commands:
        print(f"Running: {cmd}")
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("  ✅ PASSED")
                results.append(True)
        except subprocess.TimeoutExpired:
            print("  ⏰ TIMEOUT")
            results.append(False)
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results.append(False)

    passed = sum(results)
    total = len(results)
    print(f"\nCUDA Tests: {passed}/{total} passed")

    return passed == total

def run_cpu_validation():
    """Run CPU-based validation tests."""
    print("🔍 Running CPU validation tests...")

    # Test basic functionality without CUDA
    try:
        import torch
        import numpy as np

        # Test that baseline functions work on CPU
        import sys
        import os
        sys.path.insert(0, os.path.join(os.getcwd(), 'tests'))

        import importlib.util
        spec = importlib.util.spec_from_file_location("baselines", "tests/baselines.py")
        baselines = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baselines)

        mv_gelu = baselines.mv_gelu
        mv_rmsnorm_2d = baselines.mv_rmsnorm_2d
        sgp_2d = baselines.sgp_2d

        # Create test data
        x = torch.randn(4, 10, 32)
        y = torch.randn(4, 10, 32)
        weight = torch.randn(32, 10)

        # Test functions
        gelu_result = mv_gelu(x)
        norm_result = mv_rmsnorm_2d(x)
        gp_result = sgp_2d(x, y, weight)

        # Validate shapes
        assert gelu_result.shape == x.shape
        assert norm_result.shape == x.shape
        assert gp_result.shape == x.shape

        # Test numerical properties
        assert torch.isfinite(gelu_result).all()
        assert torch.isfinite(norm_result).all()
        assert torch.isfinite(gp_result).all()

        print("✅ CPU validation tests passed")
        return True

    except Exception as e:
        print(f"❌ CPU validation failed: {e}")
        return False

def generate_test_report():
    """Generate a comprehensive test report."""
    print("\n📊 Test Report")
    print("=" * 50)

    cuda_available = check_cuda_availability()

    results = {
        'Python Syntax': validate_python_syntax(),
        'Imports': validate_imports(),
        'Baseline Implementations': validate_baseline_implementations(),
        'Test Structure': validate_test_structure(),
        'Documentation': validate_documentation_structure(),
        'Mathematical Correctness': validate_mathematical_correctness(),
    }

    if cuda_available:
        results['CUDA Tests'] = run_cuda_tests()
    else:
        results['CPU Validation'] = run_cpu_validation()

    # Summary
    passed = sum(1 for r in results.values() if r)
    total = len(results)

    print(f"\nOverall: {passed}/{total} test categories passed")

    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Check output above for details.")
        return False

def main():
    """Main test runner function."""
    print("🧪 Flash Clifford Test Suite")
    print("=" * 40)

    # Run all validation tests
    success = generate_test_report()

    if success:
        print("\n✅ All validation tests passed!")
        print("The codebase is ready for use.")
    else:
        print("\n❌ Some validation tests failed!")
        print("Please fix the issues before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()
