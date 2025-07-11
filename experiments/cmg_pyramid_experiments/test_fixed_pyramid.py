#!/usr/bin/env python3
"""
Quick test script for the fixed CMG-U-Net Pyramid
"""

# Test the fixed version
import torch
from cmg_unet_pyramid import test_pyramid_architecture

print("🔧 Testing Fixed CMG-U-Net Pyramid...")
print("=" * 40)

try:
    model, outputs = test_pyramid_architecture()
    
    if outputs is not None:
        print(f"\n✅ SUCCESS! Architecture is working correctly")
        print(f"   Output shape: {outputs['logits'].shape}")
        print(f"   Phi-gamma loss: {outputs['phi_gamma_loss'].item():.4f}")
        print(f"   Number of pyramid levels: {len(outputs['assignment_matrices'])}")
        
        # Check pyramid structure
        print(f"\n📊 Pyramid Structure Analysis:")
        for i, P in enumerate(outputs['assignment_matrices']):
            cluster_sizes = P.sum(dim=0).detach().cpu().numpy()
            print(f"   Level {i}: {P.shape[0]} → {P.shape[1]} (actual clusters with nodes: {(cluster_sizes > 0.1).sum()})")
        
        print(f"\n🎯 Ready for full training!")
        
    else:
        print("❌ Test failed - check the error above")
        
except Exception as e:
    print(f"❌ Test error: {e}")
    import traceback
    traceback.print_exc()