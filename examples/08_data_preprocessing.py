#!/usr/bin/env python3
"""
Data Preprocessing Example for Flash Clifford

This example demonstrates preprocessing data for Clifford neural networks,
including multivector encoding and data loading utilities.

Output: Saves results to output/data_preprocessing/
"""

import torch
import numpy as np
import os

# Create output directory
output_dir = "output/data_preprocessing"
os.makedirs(output_dir, exist_ok=True)

print("📊 Flash Clifford - Data Preprocessing Example")
print("=" * 49)

class MultivectorEncoder:
    """Utility class for encoding data as multivectors."""

    def __init__(self, dims=2):
        self.dims = dims
        self.mv_dim = 4 if dims == 2 else 8

    def encode_2d_points(self, points):
        """Encode 2D points as multivectors."""
        # points shape: (batch, n_points, 2) -> (batch, n_points, 4)
        batch_size, n_points, _ = points.shape

        multivectors = torch.zeros(batch_size, n_points, self.mv_dim)

        # Scalar component: 1.0
        multivectors[:, :, 0] = 1.0

        # Vector components: x, y coordinates
        multivectors[:, :, 1] = points[:, :, 0]  # x component
        multivectors[:, :, 2] = points[:, :, 1]  # y component

        # Pseudoscalar component: 0.0 (for pure vectors)
        multivectors[:, :, 3] = 0.0

        return multivectors

    def encode_3d_points(self, points):
        """Encode 3D points as multivectors."""
        # points shape: (batch, n_points, 3) -> (batch, n_points, 8)
        batch_size, n_points, _ = points.shape

        multivectors = torch.zeros(batch_size, n_points, self.mv_dim)

        # Scalar component: 1.0
        multivectors[:, :, 0] = 1.0

        # Vector components: x, y, z coordinates
        multivectors[:, :, 1] = points[:, :, 0]  # x component
        multivectors[:, :, 2] = points[:, :, 1]  # y component
        multivectors[:, :, 3] = points[:, :, 2]  # z component

        # Bivector components: 0.0 (for pure vectors)
        multivectors[:, :, 4] = 0.0  # xy
        multivectors[:, :, 5] = 0.0  # xz
        multivectors[:, :, 6] = 0.0  # yz

        # Pseudoscalar component: 0.0
        multivectors[:, :, 7] = 0.0

        return multivectors

    def encode_vectors(self, vectors):
        """Encode vector data as multivectors."""
        if self.dims == 2:
            return self.encode_2d_points(vectors)
        else:
            return self.encode_3d_points(vectors)

def create_synthetic_dataset():
    """Create synthetic dataset for demonstration."""
    print("Generating synthetic dataset...")

    # Create 2D dataset
    n_samples_2d = 1000
    n_points_2d = 50

    # Generate random 2D points
    points_2d = torch.randn(n_samples_2d, n_points_2d, 2) * 10

    # Generate labels (e.g., clustering task)
    centers = torch.tensor([[5, 5], [-5, 5], [5, -5], [-5, -5]], dtype=torch.float32)
    labels_2d = torch.argmin(torch.cdist(points_2d.mean(dim=1), centers), dim=1)

    # Create 3D dataset
    n_samples_3d = 800
    n_points_3d = 40

    # Generate random 3D points
    points_3d = torch.randn(n_samples_3d, n_points_3d, 3) * 8

    # Generate labels for 3D
    centers_3d = torch.tensor([
        [4, 4, 4], [-4, 4, 4], [4, -4, 4], [-4, -4, 4],
        [4, 4, -4], [-4, 4, -4], [4, -4, -4], [-4, -4, -4]
    ], dtype=torch.float32)
    labels_3d = torch.argmin(torch.cdist(points_3d.mean(dim=1), centers_3d), dim=1)

    return {
        '2d': {
            'points': points_2d,
            'labels': labels_2d,
            'n_classes': len(centers)
        },
        '3d': {
            'points': points_3d,
            'labels': labels_3d,
            'n_classes': len(centers_3d)
        }
    }

try:
    print("🚀 Starting data preprocessing example...")

    # Create encoder
    encoder_2d = MultivectorEncoder(dims=2)
    encoder_3d = MultivectorEncoder(dims=3)

    print("✅ Created multivector encoders")

    # Generate synthetic dataset
    dataset = create_synthetic_dataset()

    print("✅ Generated synthetic dataset:")
    print(f"   - 2D dataset: {dataset['2d']['points'].shape}")
    print(f"   - 3D dataset: {dataset['3d']['points'].shape}")

    # Encode data as multivectors
    print("\n🔄 Encoding data as multivectors...")

    mv_2d = encoder_2d.encode_vectors(dataset['2d']['points'])
    mv_3d = encoder_3d.encode_vectors(dataset['3d']['points'])

    print("✅ Encoded multivectors:")
    print(f"   - 2D multivectors: {mv_2d.shape}")
    print(f"   - 3D multivectors: {mv_3d.shape}")

    # Verify encoding correctness
    print("\n🔍 Verifying encoding correctness...")

    # Check that scalar components are 1.0
    assert torch.allclose(mv_2d[:, :, 0], torch.ones_like(mv_2d[:, :, 0])), "2D scalar component incorrect"
    assert torch.allclose(mv_3d[:, :, 0], torch.ones_like(mv_3d[:, :, 0])), "3D scalar component incorrect"

    # Check that vector components match input
    assert torch.allclose(mv_2d[:, :, 1:3], dataset['2d']['points']), "2D vector components incorrect"
    assert torch.allclose(mv_3d[:, :, 1:4], dataset['3d']['points']), "3D vector components incorrect"

    print("✅ Encoding correctness verified")

    # Test data loading and batching
    print("\n📦 Testing data loading and batching...")

    batch_size = 32

    # Create data loader for 2D data
    dataset_2d_tensor = torch.utils.data.TensorDataset(
        mv_2d, dataset['2d']['labels']
    )
    dataloader_2d = torch.utils.data.DataLoader(
        dataset_2d_tensor, batch_size=batch_size, shuffle=True
    )

    # Create data loader for 3D data
    dataset_3d_tensor = torch.utils.data.TensorDataset(
        mv_3d, dataset['3d']['labels']
    )
    dataloader_3d = torch.utils.data.DataLoader(
        dataset_3d_tensor, batch_size=batch_size, shuffle=True
    )

    # Test a few batches
    batch_2d = next(iter(dataloader_2d))
    batch_3d = next(iter(dataloader_3d))

    print("✅ Data loading working:")
    print(f"   - 2D batch: {batch_2d[0].shape}, labels: {batch_2d[1].shape}")
    print(f"   - 3D batch: {batch_3d[0].shape}, labels: {batch_3d[1].shape}")

    # Save preprocessing results
    preprocessing_results = {
        'dataset_info': {
            '2d_samples': dataset['2d']['points'].shape[0],
            '3d_samples': dataset['3d']['points'].shape[0],
            '2d_classes': dataset['2d']['n_classes'],
            '3d_classes': dataset['3d']['n_classes'],
        },
        'encoding_results': {
            '2d_multivectors_shape': mv_2d.shape,
            '3d_multivectors_shape': mv_3d.shape,
            'scalar_components_correct': torch.allclose(mv_2d[:, :, 0], torch.ones_like(mv_2d[:, :, 0])),
            'vector_components_correct': torch.allclose(mv_2d[:, :, 1:3], dataset['2d']['points']),
        },
        'batching_info': {
            'batch_size': batch_size,
            '2d_batch_shape': batch_2d[0].shape,
            '3d_batch_shape': batch_3d[0].shape,
        },
        'sample_data': {
            '2d_sample_points': dataset['2d']['points'][0, :3].tolist(),
            '2d_sample_multivector': mv_2d[0, :3].tolist(),
            '3d_sample_points': dataset['3d']['points'][0, :3].tolist(),
            '3d_sample_multivector': mv_3d[0, :3].tolist(),
        }
    }

    torch.save(preprocessing_results, f"{output_dir}/preprocessing_results.pt")
    print(f"\n💾 Preprocessing results saved to: {output_dir}/preprocessing_results.pt")

    print("\n🎉 Data preprocessing example completed successfully!")
    print("✅ Multivector encoding working correctly")
    print("✅ Data loading and batching functional")
    print("✅ Encoding correctness verified")
    print("✅ Both 2D and 3D data supported")

except Exception as e:
    print(f"❌ Error during execution: {e}")
    raise

print("\n" + "=" * 49)
print("Example completed. Check output/data_preprocessing/ for results.")
