import torch

def ctc_collate(batch):
    images = []
    labels = []
    label_lengths = []

    for img, label in batch:
        images.append(img)
        labels.append(label)
        label_lengths.append(len(label))

    images = torch.stack(images)
    labels = torch.cat(labels)
    label_lengths = torch.IntTensor(label_lengths)

    return images, labels, label_lengths
