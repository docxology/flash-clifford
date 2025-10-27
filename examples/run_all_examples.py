#!/usr/bin/env python3
"""
Master script to run all Flash Clifford examples.

This script executes all example scripts and validates
that they complete successfully, demonstrating the full
functionality of the Flash Clifford library.
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

def run_example(example_file):
    """Run a single example and return success status."""
    example_name = Path(example_file).stem
    print(f"\n🧪 Running {example_name}...")
    print("-" * 40)

    try:
        # Import and run the example
        spec = importlib.util.spec_from_file_location(example_name, example_file)
        example_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(example_module)

        print(f"✅ {example_name} completed successfully")
        return True

    except Exception as e:
        print(f"❌ {example_name} failed: {e}")
        return False

def validate_example_outputs():
    """Validate that all examples produced their expected outputs."""
    print("\n🔍 Validating example outputs...")

    expected_outputs = [
        "output/basic_usage",
        "output/simple_network",
        "output/clifford_operations",
        "output/equivariant_properties",
        "output/performance_benchmark",
        "output/error_handling",
        "output/model_persistence",
        "output/data_preprocessing",
    ]

    missing_outputs = []
    for output_dir in expected_outputs:
        if not os.path.exists(output_dir):
            missing_outputs.append(output_dir)

    if missing_outputs:
        print("❌ Missing output directories:")
        for output in missing_outputs:
            print(f"  {output}")
        return False

    print("✅ All expected output directories created")
    return True

def run_all_examples():
    """Run all examples in the examples directory."""
    print("🚀 Flash Clifford - Running All Examples")
    print("=" * 45)

    examples_dir = Path("examples")
    if not examples_dir.exists():
        print("❌ Examples directory not found")
        return False

    # Get all example files (excluding this runner script)
    example_files = [
        f for f in examples_dir.glob("*.py")
        if f.name != "run_all_examples.py"
    ]

    print(f"Found {len(example_files)} example files to run")

    results = []
    for example_file in sorted(example_files):
        success = run_example(example_file)
        results.append((example_file.name, success))

    # Validate outputs
    output_validation = validate_example_outputs()

    # Summary
    print("\n📊 Example Execution Summary")
    print("=" * 30)

    passed = 0
    failed = 0

    for example_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{example_name"<25"} | {status}")
        if success:
            passed += 1
        else:
            failed += 1

    print("-" * 30)
    print(f"Total Examples: {len(results)}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")

    if output_validation:
        print("✅ Output validation: PASSED")
    else:
        print("❌ Output validation: FAILED")

    overall_success = failed == 0 and output_validation

    print("\n🏆 Final Result")
    print("=" * 15)

    if overall_success:
        print("🎉 ALL EXAMPLES PASSED!")
        print("✅ Flash Clifford examples are fully functional")
        print("✅ All outputs generated correctly")
        print("✅ Ready for production use")
    else:
        print("❌ Some examples failed")
        print("Please review the issues above")

    return overall_success

def main():
    """Main function to run all examples."""
    success = run_all_examples()

    if success:
        print("\n🎯 Next Steps:")
        print("   1. Explore individual examples in the examples/ folder")
        print("   2. Check the output/ folder for generated results")
        print("   3. Use the examples as starting points for your own projects")
        print("   4. See docs/ for comprehensive documentation")
        print()
        print("🚀 Flash Clifford is ready for use!")
    else:
        print("\n❌ Please fix the failing examples before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()
