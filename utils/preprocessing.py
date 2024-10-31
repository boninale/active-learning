import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.transforms import v2
from PIL import Image, UnidentifiedImageError

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        if self.transform:
            img = self.transform(img_path)
            return img, idx  # Return the image and its index
        return img_path, idx  # Return the image and its index

def create_training_generators(datapath, val_path = None, batch_size=64, val_split=0.2, IMG_SIZE = (224, 224), num_workers = 6):
    """Create training and validation data generators."""
    transform = transforms.Compose([
        v2.Resize(size=IMG_SIZE, antialias=True),
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