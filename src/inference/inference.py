import torch
import cv2
from src.init import preprocess_image, decode_ctc, load_model
from src.model import OCRNet

# Load character mappings (ensure CHARS matches your dataset)
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
char_to_index = {char: idx + 1 for idx, char in enumerate(CHARS)}  # Leave 0 for blank (CTC)
index_to_char = {idx: char for char, idx in char_to_index.items()}

# Path to trained model weights
MODEL_PATH = "ocr_model.pth"

# Load pre-trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(char_to_index) + 1  # Include blank index for CTC loss
model = OCRNet(num_classes=num_classes)
model = load_model(model, MODEL_PATH, device)

def infer(image_path):
    """
    Perform OCR on a single image.
    Args:
        image_path: Path to the input image.

    Returns:
        Decoded text from the image.
    """
    # Preprocess image
    img_tensor = preprocess_image(image_path).to(device)
    
    # Forward pass through the model
    with torch.no_grad():
        output = model(img_tensor)  # Output shape: [batch_size x time_steps x num_classes]
        output = output.permute(1, 0, 2)  # Reshape for CTC decoding
    
    # Decode output into text
    decoded_texts = decode_ctc(output)
    
    return decoded_texts[0]  # Return decoded text for the first image

# Test inference on a sample image
sample_image_path = "path/to/sample/image.jpg"  # Replace with actual path to an image
decoded_text = infer(sample_image_path)
print(f"Decoded Text: {decoded_text}")
