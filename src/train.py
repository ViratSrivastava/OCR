import torch
from torch.utils.data import DataLoader
from data_processing.dataset import OCRDataset  # Adjusted import based on your structure
from models.nn_char_recog import OCRNet
from init import char_to_index  # Assuming char_to_index is defined in init.py

# Custom collate function for CTC loss
def ctc_collate(batch):
    images = []
    labels = []
    label_lengths = []

    for img, label in batch:
        images.append(img)
        labels.extend(label.tolist())
        label_lengths.append(len(label))

    images = torch.stack(images)
    labels = torch.IntTensor(labels)
    label_lengths = torch.IntTensor(label_lengths)

    return images, labels, label_lengths

# Configurations
DATASET_PATH = "data/dataset_fixed"
BATCH_SIZE = 64
NUM_EPOCHS = 150
LR = 0.0003

# Prepare datasets and loaders
train_dataset = OCRDataset(root=DATASET_PATH, split='train', charset=char_to_index)
val_dataset = OCRDataset(root=DATASET_PATH, split='val', charset=char_to_index)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=ctc_collate)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=ctc_collate)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = OCRNet(num_classes=len(char_to_index) + 1).to(device)

criterion = torch.nn.CTCLoss(blank=0)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_loss = 0
    
    for images, labels, label_lengths in train_loader:
        images = images.to(device)
        
        # Debug input shape
        print(f"Input shape: {images.shape}")
        
        # Forward pass
        outputs = model(images)
        
        # Debug output shape
        if outputs is None:
            raise ValueError("Model returned None. Check the forward() method in your OCRNet class.")
        print(f"Model output shape: {outputs.shape}")
        
        input_lengths = torch.full(size=(outputs.size(0),), fill_value=outputs.size(1), dtype=torch.long)
        
        loss = criterion(
            outputs.permute(1, 0, 2), labels.to(device), input_lengths.to(device), label_lengths.to(device)
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}")
