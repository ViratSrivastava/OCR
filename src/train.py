import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
from models.nn import CRNN  # Import your model

# Define Dataset Class
class OCRDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        img = cv2.resize(img, (128, 32))  # Resize for consistency
        img = np.expand_dims(img, axis=0)  # Add channel dimension
        img = torch.FloatTensor(img) / 255.0  # Normalize

        label = self.image_files[idx].split('.')[0]  # Extract label from filename

        return img, label

# Define Training Function
def train_model(data_path, epochs=10, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(num_classes=37).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CTCLoss()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = OCRDataset(os.path.join(data_path, 'train'), transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images = images.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # Forward pass

            # Compute loss (dummy target length for now)
            target_lengths = torch.IntTensor([len(label) for label in labels])
            log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
            input_lengths = torch.full((batch_size,), outputs.size(0), dtype=torch.int32)
            loss = criterion(log_probs, labels, input_lengths, target_lengths)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

        # Save model after each epoch
        torch.save(model.state_dict(), f'weights/ocr_epoch_{epoch+1}.pth')

    print("Training Completed. Model saved in weights/")

if __name__ == "__main__":
    train_model("data", epochs=10)
