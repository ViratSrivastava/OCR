import os
import shutil

# Paths to original dataset and annotations
original_data_path = 'data\\mjsynth\\90kDICT32px'  # Path to your original dataset
new_data_path = 'data\\dataset'  # Path where you want to organize the dataset

# Create new directories for train, val, and test splits
splits = ['train', 'val', 'test']
for split in splits:
    os.makedirs(os.path.join(new_data_path, split, 'images'), exist_ok=True)

# Function to process annotations and move images
def process_annotations(split):
    annotation_file = f'annotation_{split}.txt'
    with open(os.path.join(original_data_path, annotation_file), 'r') as f:
        for line in f:
            img_path, label = line.strip().split(' ')
            img_name = os.path.basename(img_path)  # Extract image filename
            
            # Copy image to new location
            src_path = os.path.join(original_data_path, img_path)
            dest_path = os.path.join(new_data_path, split, 'images', img_name)
            shutil.copy(src_path, dest_path)
            
            # Append label to labels.txt
            with open(os.path.join(new_data_path, split, 'labels.txt'), 'a') as label_file:
                label_file.write(f"{img_name}\t{label}\n")

# Process each split (train, val, test)
for split in splits:
    print(f"Processing {split} split...")
    process_annotations(split)
print("Dataset organization complete!")
