"""
Image Preprocessing and Data Loading Script

This script provides functionality for loading, preprocessing, and creating data loaders for image datasets.
It includes an optimized Dataset class for handling images, custom collate functions, and functions for creating
training and validation data loaders.

Functions:
- ImageDataset: Custom Dataset class for loading and preprocessing images.
- custom_collate: Custom collate function to handle batches with None values.
- create_training_generators: Function to create training and validation data loaders.
- preprocess_image: Function to preprocess a single image.

Usage:
1. Define the source directory, validation directory (optional), and other parameters.
2. Create data loaders using the create_training_generators function.
3. Use the ImageDataset class for custom data loading and preprocessing.
4. Preprocess individual images using the preprocess_image function.

"""

import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.transforms import v2
from PIL import Image, UnidentifiedImageError

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define optimized Dataset class
class ImageDataset(Dataset):
    def __init__(self, img_list, data_path, device, img_size=(224, 224)):
        self.img_list = img_list
        self.data_path = data_path
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.data_path, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image).to(self.device)  # Preprocess directly on GPU
            return img_name, image
        
        except UnidentifiedImageError:
            print(f"Skipping corrupted image: {img_path}")
            return None
        
def custom_collate(batch):
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    img_names, img_tensors = zip(*batch)
    img_tensors = torch.stack(img_tensors, dim=0)
    return img_names, img_tensors
        
def create_training_generators(datapath, val_path = None, batch_size=64, val_split=0.2, IMG_SIZE = (224, 224), num_workers = 6):
    """Create training and validation data generators."""
    transform = transforms.Compose([
          v2.RandomHorizontalFlip(p=0.5),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the entire dataset
    full_dataset = datasets.ImageFolder(root=datapath, transform=transform)
    
    if val_path==None: 
        # Calculate the number of samples for training and validation
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        
        # Split the dataset
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    else:  
        train_dataset = full_dataset
        val_dataset = datasets.ImageFolder(root=val_path, transform=transform)
        print(f'Using validation set from {val_path}')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    
    return train_loader, val_loader

# Load and preprocess new images
def preprocess_image(image_path, IMG_SIZE = (224, 224)):

    # Define the image preprocessing function
    preprocess = transforms.Compose([
        v2.Resize(size=IMG_SIZE, antialias=True).to(device),
        v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]).to(device),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)
    ])

    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = preprocess(img)

        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        img_tensor = img_tensor.to(device)
        return img_tensor
    
    except UnidentifiedImageError:
        print(f"Skipping corrupted image: {image_path}")
        return None