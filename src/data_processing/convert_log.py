import os
from tqdm import tqdm

base_dir = "data/dataset_fixed"
category = "train"
img_dir = os.path.join(base_dir, category, "images")
labels_file = os.path.join(base_dir, category, "labels.txt")

# Get image files
img_files = set(os.listdir(img_dir))
print(f"Found {len(img_files)} images in {img_dir}")

# Filter labels
valid_lines = []
with open(labels_file, 'r') as f:
    for line in tqdm(f, desc="Checking labels"):
        stripped = line.strip()
        if not stripped:
            continue
        filename, label = stripped.split('\t')
        if filename in img_files:
            valid_lines.append(line)
print(f"Found {len(valid_lines)} valid label-image pairs")

# Write cleaned labels.txt
with open(labels_file, 'w') as f:
    f.write("".join(valid_lines))