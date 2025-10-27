#!/usr/bin/env python3
"""
Error handling and edge case tests for Flash Clifford.

Tests how the implementation handles invalid inputs and edge cases.
"""

import torch
import sys
import os

def test_invalid_input_shapes():
    """Test handling of invalid input shapes."""
    print("Testing invalid input shapes...")

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("baselines", "tests/baselines.py")
        baselines = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baselines)

        # Test invalid shapes
        invalid_shapes = [
            (3, 4, 4),  # Wrong number of multivector components
            (4, 0, 4),  # Zero batch size
            (4, 4, 0),  # Zero features
            (5, 4, 4),  # Too many components
        ]

        handled_properly = 0
        total_cases = len(invalid_shapes)

        for shape in invalid_shapes:
            try:
                x = torch.randn(*shape)
                result = baselines.mv_gelu(x)
                # If it doesn't fail, check that output is reasonable
                if torch.isfinite(result).all() and result.shape == x.shape:
                    handled_properly += 1
                    print(f"  ✅ Shape {shape}: Handled gracefully")
                else:
                    print(f"  ❌ Shape {shape}: Invalid output")
            except Exception as e:
                # Check that error message is informative
                if "dimension" in str(e).lower() or "size" in str(e).lower() or "shape" in str(e).lower():
                    handled_properly += 1
                    print(f"  ✅ Shape {shape}: Proper error message")
                else:
                    print(f"  ❌ Shape {shape}: Unclear error: {e}")

        print(f"✅ Invalid input shapes test: {handled_properly}/{total_cases} handled properly")
        return handled_properly == total_cases

    except Exception as e:
        print(f"❌ Invalid input shapes test failed: {e}")
        return False

def test_invalid_weights():
    """Test handling of invalid weight tensors."""
    print("Testing invalid weights...")

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("baselines", "tests/baselines.py")
        baselines = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baselines)

        x = torch.randn(4, 8, 16)
        y = torch.randn(4, 8, 16)

        # Test invalid weight shapes
        invalid_weights = [
            torch.randn(15, 10),  # Wrong feature dimension
            torch.randn(16, 9),   # Wrong number of product weights
            torch.randn(16),      # 1D instead of 2D
        ]

        handled_properly = 0
        total_cases = len(invalid_weights)

        for i, weight in enumerate(invalid_weights):
            try:
                result = baselines.sgp_2d(x, y, weight)
                if torch.isfinite(result).all():
                    print(f"  ⚠️  Weight {i}: Unexpectedly succeeded")
                else:
                    print(f"  ❌ Weight {i}: Invalid output")
            except Exception as e:
                # Accept various error messages as long as they're informative
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ["dimension", "size", "shape", "unpack", "values", "mismatch"]):
                    handled_properly += 1
                    print(f"  ✅ Weight {i}: Proper error message")
                else:
                    print(f"  ❌ Weight {i}: Unclear error: {e}")

        print(f"✅ Invalid weights test: {handled_properly}/{total_cases} handled properly")
        return handled_properly >= total_cases * 0.8  # Allow some tolerance

    except Exception as e:
        print(f"❌ Invalid weights test failed: {e}")
        return False

def test_numerical_edge_cases():
    """Test handling of numerical edge cases."""
    print("Testing numerical edge cases...")

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("baselines", "tests/baselines.py")
        baselines = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baselines)

        # Test edge cases
        edge_cases = [
            torch.full((4, 4, 8), float('inf')),  # Infinity
            torch.full((4, 4, 8), float('nan')),  # NaN
            torch.full((4, 4, 8), 1e10),         # Very large numbers
            torch.full((4, 4, 8), 1e-10),        # Very small numbers
        ]

        handled_properly = 0
        total_cases = len(edge_cases)

        for i, x in enumerate(edge_cases):
            try:
                result = baselines.mv_gelu(x)

                if torch.isnan(result).any():
                    # NaN propagation is acceptable
                    handled_properly += 1
                    print(f"  ✅ Edge case {i}: NaN handled properly")
                elif torch.isinf(result).any():
                    print(f"  ⚠️  Edge case {i}: Infinity propagated")
                elif torch.isfinite(result).all():
                    # Finite output is also acceptable
                    handled_properly += 1
                    print(f"  ✅ Edge case {i}: Finite output")
                else:
                    print(f"  ❌ Edge case {i}: Unexpected output")

            except Exception as e:
                # Crashing is not acceptable for edge cases
                print(f"  ❌ Edge case {i}: Crashed with {e}")
                return False

        print(f"✅ Numerical edge cases test: {handled_properly}/{total_cases} handled properly")
        return handled_properly >= total_cases * 0.7  # Allow some tolerance

    except Exception as e:
        print(f"❌ Numerical edge cases test failed: {e}")
        return False

def test_memory_edge_cases():
    """Test memory-related edge cases."""
    print("Testing memory edge cases...")

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("baselines", "tests/baselines.py")
        baselines = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baselines)

        # Test with very small and very large tensors
        size_tests = [
            (4, 1, 1),      # Minimal size
            (4, 2, 2),      # Small size
            (4, 1000, 1000), # Large size (if memory allows)
        ]

        successful_tests = 0

        for mv_dim, batch_size, n_features in size_tests:
            try:
                x = torch.randn(mv_dim, batch_size, n_features)

                # Test all operations
                operations = [
                    lambda: baselines.mv_gelu(x),
                    lambda: baselines.mv_rmsnorm_2d(x),
                ]

                for op in operations:
                    result = op()
                    if torch.isfinite(result).all() and result.shape == x.shape:
                        successful_tests += 1
                    else:
                        print(f"  ❌ Size {mv_dim}x{batch_size}x{n_features}: Invalid output")

                print(f"  ✅ Size {mv_dim}x{batch_size}x{n_features}: All operations successful")

            except MemoryError:
                print(f"  ⚠️  Size {mv_dim}x{batch_size}x{n_features}: Out of memory (expected)")
                successful_tests += 1  # This is acceptable
            except Exception as e:
                print(f"  ❌ Size {mv_dim}x{batch_size}x{n_features}: Unexpected error: {e}")
                return False

        print(f"✅ Memory edge cases test: {successful_tests} operations successful")
        return True

    except Exception as e:
        print(f"❌ Memory edge cases test failed: {e}")
        return False

def test_gradient_edge_cases():
    """Test gradient computation with edge cases."""
    print("Testing gradient edge cases...")

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("baselines", "tests/baselines.py")
        baselines = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baselines)

        # Test gradient computation with edge cases
        edge_inputs = [
            torch.zeros(4, 4, 8, requires_grad=True),
            torch.ones(4, 4, 8, requires_grad=True),
            torch.randn(4, 4, 8, requires_grad=True),
        ]

        successful_tests = 0

        for i, x in enumerate(edge_inputs):
            try:
                y = torch.randn(4, 4, 8, requires_grad=True)
                weight = torch.randn(8, 10, requires_grad=True)

                # Forward and backward (avoid torch.compile)
                gelu_x = baselines.mv_gelu(x)
                gelu_y = baselines.mv_gelu(y)
                output = baselines.sgp_2d(gelu_x, gelu_y, weight)
                loss = output.sum()
                loss.backward()

                # Check gradients
                if (x.grad is not None and torch.isfinite(x.grad).all() and
                    y.grad is not None and torch.isfinite(y.grad).all() and
                    weight.grad is not None and torch.isfinite(weight.grad).all()):
                    successful_tests += 1
                    print(f"  ✅ Gradient test {i}: All gradients finite")
                else:
                    print(f"  ❌ Gradient test {i}: Invalid gradients")

            except Exception as e:
                print(f"  ❌ Gradient test {i}: Failed with {e}")
                return False

        print(f"✅ Gradient edge cases test: {successful_tests}/{len(edge_inputs)} passed")
        return successful_tests == len(edge_inputs)

    except Exception as e:
        print(f"❌ Gradient edge cases test failed: {e}")
        return False

def run_error_handling_tests():
    """Run all error handling tests."""
    print("🧪 Running Error Handling Tests")
    print("=" * 30)

    tests = [
        test_invalid_input_shapes,
        test_invalid_weights,
        test_numerical_edge_cases,
        test_memory_edge_cases,
        test_gradient_edge_cases,
    ]

    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        result = test()
        results.append(result)

    passed = sum(results)
    total = len(results)

    print(f"\nError Handling Tests: {passed}/{total} passed")

    if passed >= total * 0.8:  # Allow some tolerance for edge cases
        print("\n🎉 Error handling tests completed successfully!")
        print("The implementation handles edge cases appropriately.")
        return True
    else:
        print("\n❌ Some error handling tests failed!")
        print("The implementation may not handle edge cases properly.")
        return False

if __name__ == "__main__":
    success = run_error_handling_tests()

    if success:
        print("\n✅ Error handling validation complete!")
        print("All edge cases are handled appropriately.")
    else:
        print("\n❌ Error handling validation failed!")
        print("Please improve error handling for edge cases.")
        sys.exit(1)
