import torch

def ctc_collate(batch):
    """
    Custom collate function for variable-length labels in OCR datasets.
    Args:
        batch: List of tuples [(image_tensor, label_tensor), ...]

    Returns:
        images: Tensor of stacked images.
        labels: Concatenated tensor of labels.
        label_lengths: Tensor of original label lengths.
    """
    images = []
    labels = []
    label_lengths = []

    for img, label in batch:
        images.append(img)
        labels.extend(label.tolist())  # Flatten all labels into a single list
        label_lengths.append(len(label))  # Store original length of each label

    # Stack images into a single tensor
    images = torch.stack(images)

    # Convert labels and lengths to tensors
    labels = torch.IntTensor(labels)
    label_lengths = torch.IntTensor(label_lengths)

    return images, labels, label_lengths
