from PIL import Image
import sys

def apply_bnw_filter(input_image_path, output_image_path):
    # Open an image file
    with Image.open(input_image_path) as img:
        # Convert image to black and white
        bnw_img = img.convert('L')
        # Save the image
        bnw_img.save(output_image_path)
        print(f"Black and white image saved at {output_image_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python bnw-filter.py <input_image_path> <output_image_path>")
    else:
        input_image_path = sys.argv[1]
        output_image_path = sys.argv[2]
        apply_bnw_filter(input_image_path, output_image_path)

