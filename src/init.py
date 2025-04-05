import torch
import numpy as np
import cv2
import os

# Character Set (Only alphabetic characters)
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Mapping characters to indexes
char_to_index = {char: idx + 1 for idx, char in enumerate(CHARS)}  # Leave 0 for blank (CTC)
index_to_char = {idx: char for char, idx in char_to_index.items()}

def text_to_labels(text):
    """Convert text string to tensor indices."""
    return [char_to_index[char] for char in text if char in char_to_index]

def labels_to_text(indices):
    """CTC Greedy Decoder: Collapse repeated characters and remove blanks."""
    text = []
    prev_idx = -1
    for idx in indices:
        if idx != 0:  # Skip blank
            if idx != prev_idx:  # Collapse repeats
                text.append(index_to_char.get(idx, ""))
        prev_idx = idx
    return "".join(text)

def load_model(model, weight_path, device="cuda"):
    """Load model weights and set to evaluation mode."""
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model.to(device)

def preprocess_image(image_path):
    """Preprocess image for model inference."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")
    img = cv2.resize(img, (128, 32))  # Width x Height = 128x32
    img = np.expand_dims(img, axis=(0, 1))  # Shape: (1, 1, 32, 128)
    img = torch.FloatTensor(img) / 255.0  # Normalize
    return img
