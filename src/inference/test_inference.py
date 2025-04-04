import torch
from src.model import OCRNet

# Dummy test setup
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
char_to_index = {char: idx + 1 for idx, char in enumerate(CHARS)}  # Leave 0 for blank (CTC)
index_to_char = {idx: char for char, idx in char_to_index.items()}
num_classes = len(char_to_index) + 1

# Initialize model with random weights
model = OCRNet(num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create dummy input tensor (batch_size=1, channels=1, height=32, width=128)
dummy_input = torch.randn(1, 1, 32, 128).to(device)

# Forward pass through the model
with torch.no_grad():
    output = model(dummy_input)  # Output shape: [batch_size x time_steps x num_classes]
    print(f"Model Output Shape: {output.shape}")
