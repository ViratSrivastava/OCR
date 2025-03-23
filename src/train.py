import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.data_processing.dataset import OCRDataset
from src.models.nn_char_recog import OCRNet
from src.utils import ctc_collate
from src.init import char_to_index
from pathlib import Path

# Config
BATCH_SIZE = 32
NUM_EPOCHS = 50
LR = 0.0003
# Updated CHARS to include all possible characters
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Prepare dataset
train_dataset = OCRDataset('dataset', 'train', charset=char_to_index)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                         shuffle=True, collate_fn=ctc_collate)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OCRNet(num_classes=len(CHARS)+1).to(device)  # +1 for CTC blank
criterion = nn.CTCLoss(blank=0)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0

    for batch in train_loader:
        images, labels, label_lengths = batch
        images = images.to(device)

        # Forward pass
        outputs = model(images)
        input_lengths = torch.full((images.size(0),), outputs.size(1), dtype=torch.long)

        # CTC Loss calculation
        loss = criterion(
            outputs.permute(1, 0, 2),  # (T, N, C)
            labels.to(device),          # (N*S)
            input_lengths,              # (N)
            label_lengths               # (N)
        )

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {epoch_loss/len(train_loader):.4f}')

# Save model
torch.save(model.state_dict(), 'ocr_model.pth')
