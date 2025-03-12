# OCR Training Dataset

## Overview
This dataset is designed for training Optical Character Recognition (OCR) models. It contains a variety of images with corresponding text annotations to help train and evaluate OCR algorithms.

## Dataset Structure
The dataset is organized into the following directories:

- `images/`: Contains the input images for OCR.
- `annotations/`: Contains the text annotations for each image in JSON format.

## Image Data
The `images/` directory includes images in various formats (e.g., PNG, JPEG). Each image file is named uniquely.

## Annotations
The `annotations/` directory contains JSON files where each file corresponds to an image in the `images/` directory. The JSON files have the following structure:

```json
{
    "file_name": "image1.png",
    "text": "Sample text in the image"
}
```

## Usage
To use this dataset for training an OCR model, follow these steps:

1. Load the images from the `images/` directory.
2. Parse the corresponding JSON files from the `annotations/` directory to get the text annotations.
3. Use the images and annotations to train your OCR model.


## Acknowledgements
We would like to thank all contributors who helped in creating and annotating this dataset.

## Contact
For any questions or issues, please contact [your-email@example.com].
