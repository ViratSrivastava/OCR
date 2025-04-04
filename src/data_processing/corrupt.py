from PIL import Image
import os
import concurrent.futures
import tqdm

def check_image(img_path):
    """Check if an image is corrupted"""
    try:
        with Image.open(img_path) as img:
            img.verify()  # Check for corruption
        return None
    except Exception as e:
        return (img_path, str(e))

def main():
    dataset_path = "C:\\Users\\VIRAT\\Projects\\OCR\\data"
    image_paths = []
    
    # Collect all image paths first
    print("Collecting image paths...")
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    
    print(f"Found {len(image_paths)} images to check")
    
    # Process images in parallel with a progress bar
    corrupted = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        # Map the function to all image paths and process with a progress bar
        for result in tqdm.tqdm(executor.map(check_image, image_paths), total=len(image_paths), desc="Checking images"):
            if result:  # If not None, we found a corrupt image
                corrupted.append(result)
    
    # Report corrupted images
    print(f"\nFound {len(corrupted)} corrupted images:")
    for path, error in corrupted:
        print(f"Corrupt image: {path} -> {error}")

if __name__ == "__main__":
    main()