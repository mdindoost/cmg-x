#!/usr/bin/env python3
"""
Complete CMG-U-Net Pyramid Training Script
=========================================

Integrates:
- Your estimate_k_and_clusters.py for cluster estimation
- Your phi_gamma_autoencoder.py for pooling
- New CMG-U-Net Pyramid architecture
- GraphU-Net style channel pyramid
- Your coarse graph construction approach

Usage: python run_cmg_unet_pyramid_trainer.py
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import ToSparseTensor, ToUndirected, Compose
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Import your existing modules
from estimate_k_and_clusters import estimate_k
from phi_gamma_autoencoder_fixed import PhiGammaPooling, unpool

# Add current directory to path for imports
sys.path.append('.')

# -------------------------------
# Configuration
# -------------------------------
CONFIG = {
    'dataset_name': 'Cora',
    'channels': [64, 128, 256, 512],  # Channel pyramid
    'gnn_type': 'gcn',  # 'gcn' or 'gat'
    'epochs': 300,
    'learning_rate': 0.01,
    'weight_decay': 5e-4,
    'early_stopping_patience': 30,
    'phi_gamma_weight': 1e-3,
    'dropout': 0.2,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_results': True,
    'plot_training': True
}

print("🏔️  CMG-U-Net Pyramid: Multi-Scale Graph Learning")
print("=" * 55)

# -------------------------------
# Import CMG-U-Net Pyramid (assuming saved as cmg_unet_pyramid.py)
# -------------------------------
try:
    from cmg_unet_pyramid import CMGUNetPyramid, CMGUNetPyramidTrainer, create_cmg_unet_pyramid_config
    print("✅ CMG-U-Net Pyramid imported successfully")
except ImportError as e:
    print(f"❌ Error importing CMG-U-Net Pyramid: {e}")
    print("Please save the CMG-U-Net Pyramid code as 'cmg_unet_pyramid.py'")
    sys.exit(1)

# -------------------------------
# Load and Analyze Dataset
# -------------------------------
def load_and_analyze_dataset(dataset_name: str):
    """Load dataset and perform initial analysis."""
    transform = Compose([
        ToUndirected(),
        ToSparseTensor(remove_edge_index=False)
    ])
    
    dataset = Planetoid(root=f'./data/{dataset_name}', name=dataset_name, transform=transform)
    data = dataset[0]
    
    # Calculate num_classes from the dataset
    num_classes = dataset.num_classes
    
    print(f"\n📊 Dataset Analysis: {dataset_name}")
    print(f"-" * 30)
    print(f"   Nodes: {data.num_nodes:,}")
    print(f"   Edges: {data.num_edges:,}")
    print(f"   Features: {data.num_node_features}")
    print(f"   Classes: {num_classes}")
    print(f"   Train/Val/Test: {data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}")
    print(f"   Average degree: {data.num_edges / data.num_nodes:.2f}")
    print(f"   Graph density: {data.num_edges / (data.num_nodes * (data.num_nodes - 1)):.6f}")
    
    return dataset, data

# Load dataset
dataset, data = load_and_analyze_dataset(CONFIG['dataset_name'])

# -------------------------------
# Cluster Estimation & Hierarchy Design
# -------------------------------
def estimate_pyramid_hierarchy(data, config):
    """Estimate cluster hierarchy for pyramid architecture."""
    print(f"\n🔍 Estimating Pyramid Hierarchy...")
    print(f"-" * 35)
    
    # Convert to dense adjacency for cluster estimation
    adj_dense = data.adj_t.to_dense().float()
    
    # Estimate base cluster count
    cluster_info = estimate_k(data.adj_t, data.x, k=30, d=30)
    estimated_clusters = cluster_info["k"]
    
    print(f"   Spectral estimation: {estimated_clusters} clusters")
    print(f"   Lambda critical: {cluster_info['lambda_crit']:.4f}")
    print(f"   Filter energy: {cluster_info['energy']:.2f}")
    
    # Design pyramid based on estimation
    if estimated_clusters >= 20:
        # Large number of clusters - use 4 levels
        hierarchy = [
            estimated_clusters // 2,
            estimated_clusters // 4, 
            max(3, estimated_clusters // 8)
        ]
        channels = config['channels']
    elif estimated_clusters >= 10:
        # Medium number of clusters - use 3 levels
        hierarchy = [
            estimated_clusters // 2,
            max(3, estimated_clusters // 4)
        ]
        channels = config['channels'][:3]  # Use only first 3 channel levels
    else:
        # Small number of clusters - use 2 levels
        hierarchy = [max(3, estimated_clusters // 2)]
        channels = config['channels'][:2]  # Use only first 2 channel levels
    
    print(f"   Designed hierarchy: {hierarchy}")
    print(f"   Channel pyramid: {channels}")
    
    return {
        'estimated_clusters': estimated_clusters,
        'hierarchy': hierarchy,
        'channels': channels,
        'cluster_info': cluster_info
    }

# Estimate hierarchy
hierarchy_info = estimate_pyramid_hierarchy(data, CONFIG)

# -------------------------------
# Model Creation
# -------------------------------
def create_model(data, hierarchy_info, config):
    """Create CMG-U-Net Pyramid model."""
    print(f"\n🏗️  Building CMG-U-Net Pyramid...")
    print(f"-" * 35)
    
    model = CMGUNetPyramid(
        input_dim=data.num_node_features,
        num_classes=dataset.num_classes,
        channels=hierarchy_info['channels'],
        estimated_clusters=hierarchy_info['hierarchy'],
        gnn_type=config['gnn_type'],
        dropout=config['dropout'],
        phi_gamma_weight=config['phi_gamma_weight']
    ).to(config['device'])
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    return model

# Create model
model = create_model(data, hierarchy_info, CONFIG)

# -------------------------------
# Training Setup
# -------------------------------
def setup_training(model, config):
    """Setup training components."""
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    trainer = CMGUNetPyramidTrainer(model, phi_gamma_weight=config['phi_gamma_weight'])
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    return optimizer, trainer, scheduler

optimizer, trainer, scheduler = setup_training(model, CONFIG)

# Move data to device
x = data.x.to(CONFIG['device'])
y = data.y.to(CONFIG['device'])
adj_dense = data.adj_t.to_dense().float().to(CONFIG['device'])
edge_index = data.edge_index.to(CONFIG['device'])
train_mask = data.train_mask.to(CONFIG['device'])
val_mask = data.val_mask.to(CONFIG['device'])
test_mask = data.test_mask.to(CONFIG['device'])

# -------------------------------
# Training Functions
# -------------------------------
def evaluate_model(split='val'):
    """Evaluate model on given split."""
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(x, adj_dense, edge_index)
            mask = val_mask if split == 'val' else test_mask if split == 'test' else train_mask
            accuracy = trainer.evaluate(outputs, y, mask)
            return accuracy, outputs
        except Exception as e:
            print(f"Evaluation error: {e}")
            return 0.0, None

def print_model_analysis(outputs):
    """Print analysis of model structure and learning."""
    if outputs is None:
        return
        
    try:
        analysis = trainer.analyze_pyramid_structure(outputs)
        print(f"\n🔬 Model Analysis:")
        print(f"   Pyramid levels: {len(analysis['level_sizes'])}")
        print(f"   Level sizes: {analysis['level_sizes']}")
        
        for i, (size, entropy) in enumerate(zip(analysis['level_sizes'], analysis['assignment_entropy'])):
            print(f"   Level {i}: {size} clusters, assignment entropy: {entropy:.3f}")
            
    except Exception as e:
        print(f"Analysis error: {e}")

# -------------------------------
# Training Loop
# -------------------------------
print(f"\n🚀 Training CMG-U-Net Pyramid...")
print(f"-" * 35)
print(f"   Learning rate: {CONFIG['learning_rate']}")
print(f"   Phi-gamma weight: {CONFIG['phi_gamma_weight']}")
print(f"   Early stopping patience: {CONFIG['early_stopping_patience']}")
print(f"   Device: {CONFIG['device']}")
print()

# Training tracking
training_stats = {
    'train_acc': [], 'val_acc': [], 'test_acc': [],
    'total_loss': [], 'ce_loss': [], 'phi_gamma_loss': []
}

best_val_acc = 0.0
patience_counter = 0
best_model_state = None
start_time = time.time()

for epoch in range(1, CONFIG['epochs'] + 1):
    model.train()
    optimizer.zero_grad()
    
    try:
        # Forward pass
        outputs = model(x, adj_dense, edge_index)
        
        # Compute losses
        loss_dict = trainer.compute_loss(outputs, y, train_mask)
        total_loss = loss_dict['total_loss']
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Evaluate
        train_acc = trainer.evaluate(outputs, y, train_mask)
        val_acc, _ = evaluate_model('val')
        test_acc, _ = evaluate_model('test')
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Store statistics
        training_stats['train_acc'].append(train_acc)
        training_stats['val_acc'].append(val_acc)
        training_stats['test_acc'].append(test_acc)
        training_stats['total_loss'].append(total_loss.item())
        training_stats['ce_loss'].append(loss_dict['ce_loss'].item())
        training_stats['phi_gamma_loss'].append(loss_dict['phi_gamma_loss'].item())
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Progress reporting
        if epoch % 20 == 0 or epoch <= 10:
            print(f"[Epoch {epoch:03d}] "
                  f"Loss: {total_loss.item():.4f} | "
                  f"CE: {loss_dict['ce_loss'].item():.4f} | "
                  f"Φγ: {loss_dict['phi_gamma_loss'].item():.2f} | "
                  f"Train: {train_acc:.3f} | Val: {val_acc:.3f} | Test: {test_acc:.3f}")
        
        # Early stopping
        if patience_counter >= CONFIG['early_stopping_patience']:
            print(f"\n⏹️  Early stopping at epoch {epoch} (patience: {patience_counter})")
            break
            
    except Exception as e:
        print(f"Training error at epoch {epoch}: {e}")
        break

training_time = time.time() - start_time

# Load best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Final evaluation
final_train_acc = trainer.evaluate(model(x, adj_dense, edge_index), y, train_mask)
final_val_acc, final_outputs = evaluate_model('val')
final_test_acc, _ = evaluate_model('test')

# -------------------------------
# Results Analysis
# -------------------------------
print(f"\n" + "=" * 70)
print(f"🏆 CMG-U-Net Pyramid Training Complete!")
print(f"=" * 70)
print(f"   Dataset:              {CONFIG['dataset_name']}")
print(f"   Architecture:         {len(hierarchy_info['channels'])}-level pyramid")
print(f"   Channel progression:  {' → '.join(map(str, hierarchy_info['channels']))}")
print(f"   Cluster hierarchy:    {hierarchy_info['hierarchy']}")
print(f"   Training time:        {training_time:.1f}s ({epoch} epochs)")
print(f"   Total parameters:     {sum(p.numel() for p in model.parameters()):,}")

print(f"\n📊 Performance Results:")
print(f"   Train Accuracy:       {final_train_acc:.4f}")
print(f"   Val Accuracy:         {final_val_acc:.4f}")
print(f"   Test Accuracy:        {final_test_acc:.4f}")
print(f"   Best Val Accuracy:    {best_val_acc:.4f}")
print(f"   Overfitting Gap:      {final_train_acc - final_test_acc:.4f}")

# Performance analysis
baseline_random = 1.0 / dataset.num_classes
improvement = (final_test_acc - baseline_random) / baseline_random * 100

print(f"\n📈 Performance Analysis:")
print(f"   Random baseline:      {baseline_random:.4f}")
print(f"   Improvement:          {improvement:.1f}% over random")

convergence_quality = "Good" if final_train_acc - final_test_acc < 0.15 else "Overfitting detected"
print(f"   Convergence:          {convergence_quality}")

# Model structure analysis
print_model_analysis(final_outputs)

# -------------------------------
# Visualization
# -------------------------------
def plot_training_curves():
    """Plot comprehensive training analysis."""
    if not CONFIG['plot_training']:
        return
        
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs_range = range(1, len(training_stats['train_acc']) + 1)
        
        # Accuracy curves
        ax1.plot(epochs_range, training_stats['train_acc'], 'b-', label='Train', linewidth=2, alpha=0.8)
        ax1.plot(epochs_range, training_stats['val_acc'], 'r-', label='Validation', linewidth=2, alpha=0.8)
        ax1.plot(epochs_range, training_stats['test_acc'], 'g-', label='Test', linewidth=2, alpha=0.8)
        ax1.axhline(y=baseline_random, color='gray', linestyle='--', alpha=0.5, label='Random')
        ax1.set_title('CMG-U-Net Pyramid: Accuracy Progress', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Loss curves
        ax2.semilogy(epochs_range, training_stats['total_loss'], 'purple', label='Total Loss', linewidth=2, alpha=0.8)
        ax2.semilogy(epochs_range, training_stats['ce_loss'], 'orange', label='CE Loss', linewidth=2, alpha=0.8)
        ax2.set_title('Loss Curves', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss (log scale)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Phi-gamma structural loss
        ax3.semilogy(epochs_range, training_stats['phi_gamma_loss'], 'red', linewidth=2, alpha=0.8)
        ax3.set_title('CMG Structural Loss (Φγ)', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Φγ Loss (log scale)')
        ax3.grid(True, alpha=0.3)
        
        # Learning dynamics
        train_test_gap = np.array(training_stats['train_acc']) - np.array(training_stats['test_acc'])
        ax4.plot(epochs_range, train_test_gap, 'darkred', linewidth=2, alpha=0.8)
        ax4.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='Overfitting threshold')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('Generalization Gap', fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Train - Test Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if CONFIG['save_results']:
            filename = f'cmg_unet_pyramid_{CONFIG["dataset_name"].lower()}_training.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"\n📊 Training curves saved: {filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"Plotting error: {e}")

plot_training_curves()

# -------------------------------
# Save Results
# -------------------------------
def save_experimental_results():
    """Save comprehensive experimental results."""
    if not CONFIG['save_results']:
        return
        
    results = {
        'config': CONFIG,
        'hierarchy_info': hierarchy_info,
        'model_info': {
            'channels': hierarchy_info['channels'],
            'estimated_clusters': hierarchy_info['hierarchy'],
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'architecture': 'CMG-U-Net Pyramid'
        },
        'performance': {
            'train_acc': final_train_acc,
            'val_acc': final_val_acc,
            'test_acc': final_test_acc,
            'best_val_acc': best_val_acc,
            'improvement_over_random': improvement,
            'overfitting_gap': final_train_acc - final_test_acc
        },
        'training_stats': training_stats,
        'training_time': training_time,
        'total_epochs': epoch,
        'convergence_quality': convergence_quality
    }
    
    try:
        import pickle
        filename = f'cmg_unet_pyramid_{CONFIG["dataset_name"].lower()}_results.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        print(f"💾 Results saved: {filename}")
        
        # Also save model state
        model_filename = f'cmg_unet_pyramid_{CONFIG["dataset_name"].lower()}_model.pth'
        torch.save(best_model_state, model_filename)
        print(f"💾 Best model saved: {model_filename}")
        
    except Exception as e:
        print(f"Save error: {e}")

save_experimental_results()

# -------------------------------
# Comparison & Recommendations
# -------------------------------
print(f"\n📋 Performance Context:")
print(f"-" * 25)

# Typical baselines for comparison
baselines = {
    'Cora': {'GCN': 0.81, 'GAT': 0.83, 'GraphSAGE': 0.78, 'Your previous': 0.216},
    'CiteSeer': {'GCN': 0.70, 'GAT': 0.72, 'GraphSAGE': 0.68},
    'PubMed': {'GCN': 0.79, 'GAT': 0.77, 'GraphSAGE': 0.78}
}

if CONFIG['dataset_name'] in baselines:
    dataset_baselines = baselines[CONFIG['dataset_name']]
    print(f"   {CONFIG['dataset_name']} comparison:")
    for method, acc in dataset_baselines.items():
        comparison = "✓" if final_test_acc >= acc else "✗"
        diff = final_test_acc - acc
        print(f"     {method:15}: {acc:.3f} {comparison} ({diff:+.3f})")

print(f"\n💡 Next Steps:")
print(f"-" * 15)
if final_test_acc > 0.7:
    print(f"   🎉 Excellent results! Consider:")
    print(f"      - Ablation studies on pyramid levels")
    print(f"      - Testing on larger datasets")
    print(f"      - Comparing different CMG parameters")
elif final_test_acc > 0.5:
    print(f"   ✅ Good results! Try:")
    print(f"      - Fine-tuning phi-gamma weight")
    print(f"      - Experimenting with different channel progressions")
    print(f"      - Adding multi-scale supervision")
else:
    print(f"   🔧 Room for improvement:")
    print(f"      - Check if clusters make sense")
    print(f"      - Try different GNN architectures")
    print(f"      - Adjust learning rate and regularization")

print(f"\n🎯 CMG-U-Net Pyramid experiment completed!")
print(f"   Architecture successfully combines CMG theory with U-Net hierarchy")
print(f"   Ready for further experimentation and analysis")
print(f"=" * 70)