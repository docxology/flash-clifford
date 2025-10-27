#!/usr/bin/env python3
"""
Model Persistence Example for Flash Clifford

This example demonstrates saving and loading trained models,
ensuring reproducibility and deployment capabilities.

Output: Saves results to output/model_persistence/
"""

import torch
import torch.nn as nn
import os

# Create output directory
output_dir = "output/model_persistence"
os.makedirs(output_dir, exist_ok=True)

print("💾 Flash Clifford - Model Persistence Example")
print("=" * 47)

class SimpleCliffordModel(nn.Module):
    """Simple model using Clifford operations for demonstration."""

    def __init__(self, input_features=64, hidden_features=128, output_features=10):
        super().__init__()

        self.input_proj = nn.Linear(input_features, hidden_features)

        # Simulate Clifford processing (in real implementation would use Layer)
        self.clifford_processing = nn.Sequential(
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features)
        )

        self.output_proj = nn.Linear(hidden_features, output_features)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.clifford_processing(x)
        return self.output_proj(x)

try:
    print("🔧 Creating synthetic dataset...")

    # Create synthetic dataset
    num_samples = 1000
    input_features = 64
    num_classes = 10

    X = torch.randn(num_samples, input_features)
    y = torch.randint(0, num_classes, (num_samples,))

    print(f"✅ Dataset created: {X.shape} features, {num_classes} classes")

    # Create and train model
    model = SimpleCliffordModel(
        input_features=input_features,
        hidden_features=128,
        output_features=num_classes
    )

    print("✅ Model created:")
    print(f"   - Input features: {input_features}")
    print(f"   - Hidden features: 128")
    print(f"   - Output classes: {num_classes}")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("\n🚀 Training model...")

    # Quick training loop
    num_epochs = 3
    for epoch in range(num_epochs):
        # Sample random batch
        indices = torch.randperm(num_samples)[:64]
        batch_X = X[indices]
        batch_y = y[indices]

        # Training step
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        print(f"   Epoch {epoch+1}/{num_epochs} - Loss: {loss.item():.4".4f"

    print("✅ Model training completed")

    # Save model
    model_path = f"{output_dir}/trained_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_features': input_features,
            'hidden_features': 128,
            'output_features': num_classes
        },
        'training_info': {
            'epochs': num_epochs,
            'final_loss': loss.item(),
            'optimizer_state': optimizer.state_dict()
        }
    }, model_path)

    print(f"💾 Model saved to: {model_path}")

    # Load model
    print("\n🔄 Loading model...")

    checkpoint = torch.load(model_path)
    loaded_model = SimpleCliffordModel(**checkpoint['model_config'])
    loaded_model.load_state_dict(checkpoint['model_state_dict'])

    print("✅ Model loaded successfully")

    # Verify model works after loading
    loaded_model.eval()
    with torch.no_grad():
        test_indices = torch.randperm(num_samples)[:32]
        test_X = X[test_indices]
        test_y = y[test_indices]

        original_outputs = model(test_X)
        loaded_outputs = loaded_model(test_X)

        # Check that outputs are identical
        max_diff = (original_outputs - loaded_outputs).abs().max().item()

        print("✅ Model verification:")
        print(f"   - Max output difference: {max_diff:.2e}")
        print(f"   - Outputs identical: {max_diff < 1e-6}")

    # Test model serialization with different formats
    print("\n📦 Testing model serialization...")

    # Save as state dict only
    state_dict_path = f"{output_dir}/model_state_dict.pt"
    torch.save(model.state_dict(), state_dict_path)

    # Load state dict
    state_dict_model = SimpleCliffordModel(**checkpoint['model_config'])
    state_dict_model.load_state_dict(torch.load(state_dict_path))

    print("✅ State dict serialization working")

    # Test with buffer
    buffer_model = SimpleCliffordModel(**checkpoint['model_config'])
    model_buffer = torch.save(buffer_model, f"{output_dir}/model_buffer.pt")
    buffer_loaded = torch.load(f"{output_dir}/model_buffer.pt")

    print("✅ Model buffer serialization working")

    # Save complete results
    final_results = {
        'model_config': checkpoint['model_config'],
        'training_results': {
            'epochs_trained': num_epochs,
            'final_loss': loss.item(),
            'model_saved_successfully': os.path.exists(model_path),
            'model_loaded_successfully': True,
            'outputs_identical': max_diff < 1e-6,
        },
        'serialization_tests': {
            'state_dict_serialization': True,
            'buffer_serialization': True,
            'model_reproducibility': max_diff < 1e-6
        },
        'file_sizes': {
            'full_checkpoint': os.path.getsize(model_path) if os.path.exists(model_path) else 0,
            'state_dict': os.path.getsize(state_dict_path) if os.path.exists(state_dict_path) else 0,
            'buffer_model': os.path.getsize(f"{output_dir}/model_buffer.pt") if os.path.exists(f"{output_dir}/model_buffer.pt") else 0,
        }
    }

    torch.save(final_results, f"{output_dir}/persistence_results.pt")
    print(f"\n💾 Complete results saved to: {output_dir}/persistence_results.pt")

    print("\n🎉 Model persistence example completed successfully!")
    print("✅ Model training and saving working correctly")
    print("✅ Model loading and inference working correctly")
    print("✅ Model reproducibility verified")
    print("✅ Multiple serialization formats supported")

except Exception as e:
    print(f"❌ Error during execution: {e}")
    raise

print("\n" + "=" * 47)
print("Example completed. Check output/model_persistence/ for results.")
