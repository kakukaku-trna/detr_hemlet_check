#!/usr/bin/env python3
"""
Test script to verify local weights loading for ConvNeXt backbone.
"""

import argparse
import torch
import os
from pathlib import Path

def test_local_weights():
    parser = argparse.ArgumentParser('Test local weights loading')
    parser.add_argument('--weights', type=str, default='./temp/model.safetensors',
                        help='Path to local weights file')
    parser.add_argument('--backbone', type=str, default='convnext_tiny',
                        help='Backbone name')
    args = parser.parse_args()

    print(f"Testing local weights loading...")
    print(f"  Weights path: {args.weights}")
    print(f"  Backbone: {args.backbone}")

    # Check if file exists
    if not os.path.exists(args.weights):
        print(f"❌ Error: Weights file not found: {args.weights}")
        print(f"   Full path: {os.path.abspath(args.weights)}")
        return False

    file_size = os.path.getsize(args.weights) / (1024**3)
    print(f"✓ Weights file found: {file_size:.2f} GB")

    # Try to load with timm
    try:
        import timm
        print("✓ timm is installed")
    except ImportError:
        print("❌ timm is not installed. Please install it: pip install timm")
        return False

    # Try to load safetensors
    if args.weights.endswith('.safetensors'):
        try:
            from safetensors.torch import load_file
            print("✓ safetensors is installed")
        except ImportError:
            print("❌ safetensors is not installed. Please install it: pip install safetensors")
            return False

    # Load model
    print(f"\nLoading backbone model without pretrained weights...")
    try:
        model = timm.create_model(args.backbone, pretrained=False, features_only=True,
                                 out_indices=(1, 2, 3))
        print("✓ Model created successfully")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False

    # Load weights
    print(f"\nLoading weights from: {args.weights}")
    try:
        if args.weights.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(args.weights)
            print(f"✓ Loaded {len(state_dict)} tensors from safetensors")
        else:
            state_dict = torch.load(args.weights, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            print(f"✓ Loaded {len(state_dict)} tensors from .pth")

        # Try to load state dict
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"✓ State dict loaded successfully")
        if missing_keys:
            print(f"  ⚠ Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"  ⚠ Unexpected keys: {len(unexpected_keys)}")

    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test forward pass
    print(f"\nTesting forward pass...")
    try:
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output count: {len(output)}")
        for i, out in enumerate(output):
            print(f"    Output {i} shape: {out.shape}")
    except Exception as e:
        print(f"❌ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*50)
    print("✓ All tests passed! Local weights are ready to use.")
    print("="*50)
    print("\nYou can now train with:")
    print(f"  python main.py \\")
    print(f"    --backbone {args.backbone} \\")
    print(f"    --backbone_weights {args.weights} \\")
    print(f"    --num_decoder_layers 4 \\")
    print(f"    --num_queries 150 \\")
    print(f"    --epochs 50")

    return True

if __name__ == '__main__':
    success = test_local_weights()
    exit(0 if success else 1)
