import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging

# Log issues to file
logging.basicConfig(filename='conversion_errors.log', level=logging.WARNING, format='%(message)s')

def process_file(args):
    src_path, dst_path, label, category, index = args
    try:
        shutil.move(src_path, dst_path)
        return f"{category}_{index:06d}.jpg\t{label}"
    except Exception as e:
        logging.warning(f"Error processing {src_path}: {e}")
        return None

def convert_dataset(base_dir, category):
    src_img_dir = os.path.join(base_dir, category, "images")
    dst_img_dir = os.path.join(base_dir, category, "images")
    src_labels_file = os.path.join(base_dir, category, "labels.txt")
    dst_labels_file = os.path.join(base_dir, category, "labels.txt")

    if not os.path.exists(src_img_dir):
        print(f"Image directory not found: {src_img_dir}")
        return
    if not os.path.exists(src_labels_file):
        print(f"Labels file not found: {src_labels_file}")
        return

    # Read current labels
    label_map = {}
    print(f"Reading {src_labels_file}")
    with open(src_labels_file, 'r') as f:
        lines = f.readlines()
        print(f"Found {len(lines)} lines in {src_labels_file}")
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                logging.warning(f"Empty line at {i} in {src_labels_file}")
                continue
            parts = stripped.split('\t')
            if len(parts) < 2:
                logging.warning(f"Malformed line at {i} in {src_labels_file}: {stripped}")
                continue
            filename, label = parts[0], parts[1]
            label_map[filename] = label

    if not label_map:
        print(f"No valid label entries found in {src_labels_file}")
        return

    # Prepare tasks
    tasks = []
    for i, (old_name, label) in enumerate(label_map.items()):
        src_path = os.path.join(src_img_dir, old_name)
        new_name = f"{category}_{i:06d}.jpg"
        dst_path = os.path.join(dst_img_dir, new_name)
        if not os.path.exists(src_path):  # Shouldn’t happen post-pre-check, but keep
            logging.warning(f"Image not found: {src_path}")
            continue
        tasks.append((src_path, dst_path, label, category, i))

    if not tasks:
        print(f"No valid image-label pairs to process in {category}")
        return

    # Multithreaded renaming
    new_labels = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(process_file, tasks), total=len(tasks), desc=f"Renaming {category}"))
        for result in results:
            if result:
                new_labels.append(result)

    # Write updated labels.txt
    with open(dst_labels_file, 'w') as f:
        f.write("\n".join(new_labels))
    print(f"Processed {len(new_labels)} images for {category}")

if __name__ == "__main__":
    base_dir = "data/dataset_fixed"
    for category in ["train", "test", "val"]:
        convert_dataset(base_dir, category)