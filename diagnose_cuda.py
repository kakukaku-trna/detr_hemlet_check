#!/usr/bin/env python3
"""
Diagnose CUDA/GPU issues and suggest fixes.
"""
import torch
import sys

def diagnose():
    print("=" * 60)
    print("CUDA/GPU Diagnosis Report")
    print("=" * 60)

    # 1. PyTorch info
    print(f"\n📦 PyTorch Version: {torch.__version__}")
    print(f"📦 PyTorch CUDA: {torch.version.cuda}")

    # 2. CUDA available
    print(f"\n🔍 CUDA Available (reported): {torch.cuda.is_available()}")

    # 3. GPU count
    if torch.cuda.is_available():
        print(f"🔍 GPU Count: {torch.cuda.device_count()}")
        try:
            print(f"🔍 GPU Name: {torch.cuda.get_device_name(0)}")
        except:
            print("🔍 GPU Name: (failed to get)")

    # 4. Try actual GPU operations
    print("\n🧪 Testing GPU operations...")
    try:
        # Simple test
        x = torch.randn(100, 100, device='cuda')
        y = torch.randn(100, 100, device='cuda')
        z = torch.matmul(x, y)
        print("✅ GPU Operations: SUCCESS")
        return True
    except Exception as e:
        error_msg = str(e)
        print(f"❌ GPU Operations: FAILED")
        print(f"   Error: {error_msg[:200]}")

        # Analyze error
        if "NVML_SUCCESS" in error_msg or "Driver" in error_msg:
            print("\n📌 Problem: CUDA driver/library version mismatch")
            print("   Solution: Upgrade CUDA toolkit to 12.4+ or use CPU")
        elif "CUDA out of memory" in error_msg:
            print("\n📌 Problem: Insufficient GPU memory")
            print("   Solution: Reduce batch_size")
        elif "cuda is not available" in error_msg:
            print("\n📌 Problem: GPU not initialized")
            print("   Solution: Install CUDA drivers")
        else:
            print(f"\n📌 Problem: Unknown error - {type(e).__name__}")

        return False

if __name__ == '__main__':
    success = diagnose()

    print("\n" + "=" * 60)
    if success:
        print("✅ GPU is ready for training!")
        print("\nUse: --device cuda")
    else:
        print("⚠️  GPU training not available")
        print("\nUse: --device cpu (slower, for testing only)")
        print("\nTo fix:")
        print("1. Read: CUDA_DRIVER_TROUBLESHOOTING.md")
        print("2. Option A: Use CPU for now")
        print("3. Option B: Upgrade CUDA 11.4 → 12.4")
    print("=" * 60)

    sys.exit(0 if success else 1)
