import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CityScapeSegmentationDataset(Dataset):
    """Custom Dataset for Cityscapes segmentation task"""
    
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask


def get_transforms(phase='train', image_size=512):
    """Get data augmentation transforms"""
    
    if phase == 'train':
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Rotate(limit=30, p=0.5),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], is_check_shapes=False)
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], is_check_shapes=False)


def create_dataloaders(data_dir, batch_size=8, image_size=512, seed=42):
    """Create train and test dataloaders with 80-20 split"""
    
    # Get image and mask paths
    rgb_dir = os.path.join(data_dir, 'CameraRGB')
    mask_dir = os.path.join(data_dir, 'CameraMask')
    
    image_files = sorted(os.listdir(rgb_dir))
    mask_files = sorted(os.listdir(mask_dir))
    
    image_paths = [os.path.join(rgb_dir, f) for f in image_files]
    mask_paths = [os.path.join(mask_dir, f) for f in mask_files]
    
    # 80-20 train-test split with seed 42
    train_img, test_img, train_mask, test_mask = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=seed
    )
    
    # Create datasets
    train_dataset = CityScapeSegmentationDataset(
        train_img, train_mask, 
        transform=get_transforms('train', image_size)
    )
    
    test_dataset = CityScapeSegmentationDataset(
        test_img, test_mask,
        transform=get_transforms('test', image_size)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return train_loader, test_loader, train_dataset, test_dataset
