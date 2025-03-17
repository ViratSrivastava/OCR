import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paths (modify these)
original_data_path = "data\\dataset"  # Your original dataset folder
new_data_path = "data\\dataset_fixed"  # Where to save fixed dataset

def extract_label_from_filename(filename):
    """Extracts text label from filenames like '115_Lube_45484.jpg'"""
    parts = filename.split("_")
    if len(parts) >= 3:  # Ensure filename follows expected pattern
        return parts[1]  # Return middle segment (text label)
    else:
        print(f"⚠️ Bad filename format: {filename}")
        return ""

def process_image_and_label(line, split):
    """Process a single image and its corresponding label."""
    img_rel_path, _ = line.strip().split("\t")  # Get image path
    img_filename = os.path.basename(img_rel_path)
    
    # Extract text label from filename
    text_label = extract_label_from_filename(img_filename)
    if not text_label:
        return None  # Skip malformed filenames
    
    # Copy image to new structure
    src_img = os.path.join(original_data_path, split, "images", img_filename)
    dst_img = os.path.join(new_data_path, split, "images", img_filename)
    
    # Copy the image file
    shutil.copy(src_img, dst_img)
    
    return f"{img_filename}\t{text_label}\n"

def process_split(split):
    print(f"\nProcessing {split} split...")
    
    # Create new directories for fixed dataset
    os.makedirs(os.path.join(new_data_path, split, "images"), exist_ok=True)

    # Read original labels file
    label_file_path = os.path.join(original_data_path, split, "labels.txt")
    
    with open(label_file_path, "r") as f:
        lines = f.readlines()
    
    # Create new labels.txt
    new_label_file_path = os.path.join(new_data_path, split, "labels.txt")
    
    with open(new_label_file_path, "w") as label_file:
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_image_and_label, line, split): line for line in lines}
            for future in as_completed(futures):
                result = future.result()
                if result:  # Only write valid results
                    label_file.write(result)

# Process all splits
for split in ["train", "val", "test"]:
    process_split(split)

print("\n✅ Labels fixed! Verify dataset_fixed/labels.txt files.")
