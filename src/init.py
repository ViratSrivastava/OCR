import torch
import numpy as np
import cv2
import os

# Character Set (Modify based on dataset)
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Mapping characters to indexes
char_to_index = {char: idx + 1 for idx, char in enumerate(CHARS)}  # Leave 0 for blank (CTC)
index_to_char = {idx: char for char, idx in char_to_index.items()}

# Convert text to tensor indices
def text_to_labels(text):
    return [char_to_index[char] for char in text if char in char_to_index]

# Convert model output to readable text (CTC Decoding)
def labels_to_text(indices):
    text = []
    for idx in indices:
        if idx != 0:  # Ignore blank (CTC uses 0 as blank)
            text.append(index_to_char.get(idx, ""))
    return "".join(text)

# Load model weights
def load_model(model, weight_path, device="cuda"):
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model.to(device)

# Preprocess Image for Model Inference
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 32))
    img = np.expand_dims(img, axis=0)  # Add channel dimension
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = torch.FloatTensor(img) / 255.0  # Normalize
    return img
