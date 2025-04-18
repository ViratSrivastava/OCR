# OCR Model README

## Overview
This project implements an Optical Character Recognition (OCR) model using PyTorch. The model utilizes a Convolutional Neural Network (CNN) for feature extraction, followed by a Long Short-Term Memory (LSTM) network for sequence modeling, and a fully connected layer for character classification.

## Dataset
The model is trained on publicly available OCR datasets. The datasets used include:

### 1. SynthText
   - **Source:** [SynthText](https://www.robots.ox.ac.uk/~vgg/data/text/)
   - **Description:** A large-scale synthetic dataset containing images with randomly placed text in different fonts, orientations, and backgrounds.
   - **Usage:** Pretraining and improving text recognition performance on diverse backgrounds.

### 2. IAM Handwriting Database
   - **Source:** [IAM Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
   - **Description:** A dataset consisting of handwritten English text, useful for recognizing cursive and handwritten styles.
   - **Usage:** Fine-tuning the model for handwriting recognition tasks.

### 3. MJSynth (Synth90k)
   - **Source:** [MJSynth](https://www.robots.ox.ac.uk/~vgg/data/text/)
   - **Description:** A synthetic dataset containing approximately 9 million cropped word images, designed to train scene text recognition models.
   - **Usage:** Helps improve the model’s performance on printed text.

### 4. IIIT 5K-Word Dataset
   - **Source:** [IIIT 5K](https://cvit.iiit.ac.in/research/projects/cvit-projects/the-iiit-5k-word-dataset)
   - **Description:** A collection of 5,000 word images extracted from real-world scenes.
   - **Usage:** Useful for validating the model's generalization to real-world scenarios.

### 5. COCO-Text
   - **Source:** [COCO-Text](https://bgshih.github.io/cocotext/)
   - **Description:** A dataset derived from the COCO dataset, containing text instances in natural scenes.
   - **Usage:** Beneficial for scene text detection and recognition tasks.

## Preprocessing
Before training, the datasets undergo preprocessing:
- Images are resized to a fixed dimension (e.g., 128x32 for word-level recognition).
- Grayscale conversion for uniformity (if applicable).
- Normalization for faster convergence.
- Data augmentation (rotation, distortion, blurring) to enhance generalization.

## Training Details
- **Framework:** PyTorch
- **Model:** CNN + LSTM + Fully Connected Layer
- **Loss Function:** Connectionist Temporal Classification (CTC) Loss
- **Optimizer:** Adam
- **Batch Size:** 32
- **Learning Rate:** 0.001
- **Epochs:** 50

## Usage
1. Download and extract the datasets.
2. Update the dataset path in the training script.
3. Run the training script to train the OCR model.
4. Evaluate the model on test data.

## Acknowledgments
We acknowledge the creators of the above datasets for their contributions to the OCR research community.

For any queries or contributions, feel free to open an issue or submit a pull request.

