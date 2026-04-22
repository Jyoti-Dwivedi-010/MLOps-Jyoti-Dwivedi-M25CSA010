import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from sklearn.metrics import jaccard_score, f1_score
from dataloader import create_dataloaders


def calculate_miou(pred, target, num_classes=23):
    """Calculate mean Intersection over Union"""
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    # Flatten predictions and targets
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    iou_scores = []
    for class_id in range(num_classes):
        pred_mask = (pred_flat == class_id)
        target_mask = (target_flat == class_id)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        union = np.logical_or(pred_mask, target_mask).sum()
        
        if union == 0:
            continue
        iou = intersection / union
        iou_scores.append(iou)
    
    return np.mean(iou_scores) if iou_scores else 0.0


def calculate_mdice(pred, target, num_classes=23):
    """Calculate mean Dice coefficient"""
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()
    
    # Flatten predictions and targets
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    dice_scores = []
    for class_id in range(num_classes):
        pred_mask = (pred_flat == class_id)
        target_mask = (target_flat == class_id)
        
        intersection = np.logical_and(pred_mask, target_mask).sum()
        dice = (2 * intersection) / (pred_mask.sum() + target_mask.sum() + 1e-6)
        dice_scores.append(dice)
    
    return np.mean(dice_scores) if dice_scores else 0.0


class SegmentationTrainer:
    def __init__(self, model, device, num_classes=23):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=2
        )
        
        # History tracking
        self.train_losses = []
        self.val_losses = []
        self.val_miou_scores = []
        self.val_mdice_scores = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc='Training', disable=False)
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.long().to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validating', disable=False)
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.long().to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(masks.cpu().numpy())
                
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(val_loader)
        self.val_losses.append(avg_loss)
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        miou = calculate_miou(torch.tensor(all_preds), torch.tensor(all_targets), self.num_classes)
        mdice = calculate_mdice(torch.tensor(all_preds), torch.tensor(all_targets), self.num_classes)
        
        self.val_miou_scores.append(miou)
        self.val_mdice_scores.append(mdice)
        
        return avg_loss, miou, mdice
    
    def train(self, train_loader, val_loader, num_epochs=15):
        """Train the model for specified epochs"""
        best_miou = 0.0
        
        for epoch in range(num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            print(f"✓ Train Loss: {train_loss:.4f}")
            
            # Validate
            val_loss, miou, mdice = self.validate(val_loader)
            print(f"✓ Val Loss: {val_loss:.4f}, mIOU: {miou:.4f}, mDice: {mdice:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if miou > best_miou:
                best_miou = miou
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f" Best model saved! mIOU: {best_miou:.4f}")
    
    def plot_metrics(self, save_dir='plots'):
        """Plot training metrics"""
        os.makedirs(save_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Plot training loss
        axes[0].plot(self.train_losses, label='Train Loss', linewidth=2)
        axes[0].plot(self.val_losses, label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot mIOU
        axes[1].plot(self.val_miou_scores, 'g-', linewidth=2, marker='o')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('mIOU', fontsize=12)
        axes[1].set_title('Validation mIOU Score', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        # Plot mDice
        axes[2].plot(self.val_mdice_scores, 'b-', linewidth=2, marker='s')
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('mDice', fontsize=12)
        axes[2].set_title('Validation mDice Score', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Metrics plot saved to {save_dir}/training_metrics.png")
        plt.close()
    
    def save_metrics_json(self, save_dir='plots'):
        """Save metrics to JSON"""
        os.makedirs(save_dir, exist_ok=True)
        metrics = {
            'train_losses': [float(x) for x in self.train_losses],
            'val_losses': [float(x) for x in self.val_losses],
            'val_miou_scores': [float(x) for x in self.val_miou_scores],
            'val_mdice_scores': [float(x) for x in self.val_mdice_scores]
        }
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)


def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders with smaller batch size for CPU
    print("Loading dataset...")
    data_dir = '/home/m25csa010/M25CSA010_Major/Q2/MLDLOPs_2026_Major_Exam'
    batch_size = 2 if str(device) == 'cpu' else 8  # Smaller batch for CPU
    train_loader, test_loader, train_dataset, test_dataset = create_dataloaders(
        data_dir, batch_size=batch_size, image_size=256, seed=42
    )
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"Batch size: {batch_size}")
    
    # Create model (use MobileNet for faster training on CPU)
    print("Creating UNet model...")
    model = smp.Unet(
        encoder_name='mobilenet_v2',
        encoder_weights='imagenet',
        in_channels=3,
        classes=23,
        activation=None
    )
    model.to(device)
    
    # Train
    trainer = SegmentationTrainer(model, device, num_classes=23)
    print("\nStarting training...")
    num_epochs = 15
    trainer.train(train_loader, test_loader, num_epochs=num_epochs)
    
    # Plot metrics
    print("\nPlotting metrics...")
    trainer.plot_metrics('plots')
    trainer.save_metrics_json('plots')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    _, test_miou, test_mdice = trainer.validate(test_loader)
    print(f"\n{'='*60}")
    print(f"TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"mIOU:  {test_miou:.4f}")
    print(f"mDice: {test_mdice:.4f}")
    
    if test_miou > 0.48 and test_mdice > 0.48:
        print(" Both metrics are above 0.48 threshold!")
    else:
        print("  Some metrics are below 0.48 threshold")
    print(f"{'='*60}")
    
    # Save test metrics
    test_metrics = {
        'mIOU': float(test_miou),
        'mDice': float(test_mdice)
    }
    with open('test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print("\n✓ Training complete! Test metrics saved to test_metrics.json")


if __name__ == '__main__':
    main()
