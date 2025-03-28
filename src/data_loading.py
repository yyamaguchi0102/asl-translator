import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np

class ASLDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        """
        Args:
            root_dir (str): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
            train (bool): If True, loads training data, else loads test data
        """
        self.root_dir = os.path.join(root_dir, 'asl_alphabet_train' if train else 'asl_alphabet_test')
        self.transform = transform
        self.train = train
        
        # Get all image files and their labels
        self.images = []
        self.labels = []
        
        if train:
            # Training data is organized in subdirectories
            self.classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            
            # Load all image paths and labels
            for cls in self.classes:
                cls_dir = os.path.join(self.root_dir, cls)
                if os.path.isdir(cls_dir):  # Make sure it's a directory
                    for img_name in os.listdir(cls_dir):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Only load image files
                            self.images.append(os.path.join(cls_dir, img_name))
                            self.labels.append(self.class_to_idx[cls])
        else:
            # Test data is organized as individual files
            self.classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                          'nothing', 'space']
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
            
            # Load all image paths and labels
            for img_name in os.listdir(self.root_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Extract class from filename (e.g., 'A_test.jpg' -> 'A')
                    cls = img_name.split('_')[0]
                    if cls in self.class_to_idx:
                        self.images.append(os.path.join(self.root_dir, img_name))
                        self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    """
    Create data loaders for training and testing
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = ASLDataset(
        root_dir=os.path.join(data_dir, 'train'),
        transform=train_transform,
        train=True
    )

    test_dataset = ASLDataset(
        root_dir=os.path.join(data_dir, 'test'),
        transform=test_transform,
        train=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader, train_dataset.classes 