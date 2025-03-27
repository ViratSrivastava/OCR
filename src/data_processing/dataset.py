import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch

class OCRDataset(Dataset):
    def __init__(self, root, split, charset, transform=None):
        self.root = os.path.join(root, split)  # Path to train/val/test split
        self.transform = transform or T.Compose([
            T.Grayscale(),  # Convert image to grayscale
            T.Resize((32, 128)),  # Resize to standard dimensions
            T.ToTensor(),  # Convert image to tensor
            T.Normalize((0.5,), (0.5,))  # Normalize pixel values
        ])
        self.charset = charset  # Character-to-index mapping
        self.samples = self._load_samples()  # Load image-label pairs

    def _load_samples(self):
        """Load image-label pairs from labels.txt."""
        samples = []
        labels_path = os.path.join(self.root, 'labels.txt')
        with open(labels_path, 'r') as f:
            for line in f:
                img_name, label = line.strip().split('\t')
                samples.append((img_name, label))
        return samples

    def __getitem__(self, idx):
        """Return a single image and its corresponding label."""
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.root, 'images', img_name)
        
        # Load image and apply transformations
        img = Image.open(img_path)
        img_tensor = self.transform(img)
        
        # Convert label to tensor of indices
        label_tensor = torch.IntTensor([self.charset[c] for c in label if c in self.charset])
        
        return img_tensor, label_tensor

    def __len__(self):
        """Return total number of samples."""
        return len(self.samples)