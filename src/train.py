import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from src.models.nn_char_recog import OCRNet
from src.init import char_to_index
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from PIL import Image
import os
import numpy as np

# CTC decoding function
def ctc_greedy_decode(preds, blank=0, device='cpu'):
    preds = preds.argmax(dim=2).permute(1, 0)  # (N, T)
    decoded = []
    for pred in preds:
        prev = None
        seq = []
        for t in pred:
            if t != blank and t != prev:
                seq.append(t.item())
            prev = t
        decoded.append(torch.tensor(seq, device=device))
    return decoded

# Simple collate function
def simple_ctc_collate(batch):
    images = []
    labels = []
    label_lengths = []
    for img, label, length in batch:
        images.append(img)
        labels.append(label)
        label_lengths.append(length)
    images = torch.stack(images, dim=0)
    labels = torch.cat(labels, dim=0)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    return images, labels, label_lengths

# Standalone dataset class
class SimpleOCRDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.charset = char_to_index
        # Look for common image extensions
        self.image_files = []
        for ext in ('.jpg', '.jpeg', '.png'):
            self.image_files.extend([f for f in os.listdir(data_dir) if f.lower().endswith(ext)])
        if not self.image_files:
            # Check subfolders recursively
            for root, _, files in os.walk(data_dir):
                for ext in ('.jpg', '.jpeg', '.png'):
                    self.image_files.extend([os.path.join(root, f) for f in files if f.lower().endswith(ext)])
        print(f"Found {len(self.image_files)} images")
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {data_dir}. Check path or file extensions.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            img_path = os.path.join(self.data_dir, self.image_files[idx]) if os.path.isabs(self.image_files[idx]) else os.path.join(self.data_dir, self.image_files[idx])
            # Assume label is in filename (e.g., "img_abc_0.jpg" -> "abc") - adjust as needed
            label = os.path.splitext(self.image_files[idx])[0].split('_')[1]  # Customize this!
            img = Image.open(img_path).convert('L')
            img = img.resize((128, 32))
            img_tensor = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0) / 255.0
            label_tensor = torch.tensor([self.charset[c] for c in label], dtype=torch.long)
            return img_tensor, label_tensor, len(label)
        except (OSError, IOError, KeyError, IndexError):
            print(f"Skipping corrupted or invalid item at index {idx}")
            return torch.zeros((1, 32, 128)), torch.tensor([0], dtype=torch.long), 1

def main():
    # Config
    BATCH_SIZE = 1024
    NUM_EPOCHS = 10
    LR = 0.0003
    CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    NUM_WORKERS = 12

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = OCRNet(num_classes=len(CHARS)+1).to(device)
    criterion = nn.CTCLoss(blank=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler('cuda')

    # Dataset
    train_dataset = SimpleOCRDataset('data/dataset_fixed')
    print(f"Dataset size: {len(train_dataset)} images")
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=simple_ctc_collate,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    print(f"Number of batches: {len(train_loader)}")

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        correct_chars = 0
        total_chars = 0
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_bar = tqdm(train_loader, desc="Training", unit="batch")
        
        for batch_idx, (images, labels, label_lengths) in enumerate(train_bar):
            images, labels, label_lengths = images.to(device), labels.to(device), label_lengths.to(device)

            with autocast('cuda'):
                outputs = model(images)
                outputs = nn.functional.log_softmax(outputs, dim=2)
                input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long).to(device)
                loss = criterion(outputs, labels, input_lengths, label_lengths)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            train_bar.set_description(f"Training - Loss: {epoch_loss / (batch_idx + 1):.4f}")

        # Accuracy
        model.eval()
        with torch.no_grad():
            for images, labels, label_lengths in train_loader:
                images, labels, label_lengths = images.to(device), labels.to(device), label_lengths.to(device)
                with autocast('cuda'):
                    outputs = model(images)
                    outputs = nn.functional.log_softmax(outputs, dim=2)
                decoded_preds = ctc_greedy_decode(outputs, blank=0, device=device)
                start_idx = 0
                for i in range(images.size(0)):
                    label_length = label_lengths[i].item()
                    label_seq = labels[start_idx:start_idx + label_length]
                    start_idx += label_length
                    pred_seq = decoded_preds[i]
                    if label_seq.numel() > 0 and pred_seq.numel() > 0:
                        correct_chars += torch.sum(pred_seq[:min(len(pred_seq), len(label_seq))] == label_seq[:min(len(pred_seq), len(label_seq))]).item()
                    total_chars += label_seq.numel()

        epoch_loss = epoch_loss / len(train_loader)
        epoch_accuracy = correct_chars / total_chars if total_chars > 0 else 0
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}")

        # Save model
        if (epoch + 1) % 5 == 0 or epoch == NUM_EPOCHS - 1:
            torch.save(model.state_dict(), f'ocr_model_epoch_{epoch+1}.pth')
            print(f"Saved model to ocr_model_epoch_{epoch+1}.pth")

    print("Training complete.")

if __name__ == '__main__':
    main()