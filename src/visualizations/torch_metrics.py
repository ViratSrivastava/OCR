from torchmetrics import CharErrorRate, WordErrorRate

# Initialize metrics
cer_metric = CharErrorRate()
wer_metric = WordErrorRate()

# Inside the training loop
for images, labels in train_loader:
    images = images.to(device)

    optimizer.zero_grad()
    outputs = model(images)  # Forward pass

    # Compute loss
    target_lengths = torch.IntTensor([len(label) for label in labels])
    log_probs = torch.nn.functional.log_softmax(outputs, dim=2)
    input_lengths = torch.full((batch_size,), outputs.size(0), dtype=torch.int32)
    loss = criterion(log_probs, labels, input_lengths, target_lengths)

    loss.backward()
    optimizer.step()
    total_loss += loss.item()

    # Decode predictions (dummy decoding for now)
    predictions = torch.argmax(log_probs, dim=2).permute(1, 0).cpu().numpy()
    decoded_preds = ["".join([chr(p) for p in pred if p != 0]) for pred in predictions]
    decoded_labels = labels  # Assuming labels are already strings

    # Update metrics
    cer_metric.update(decoded_preds, decoded_labels)
    wer_metric.update(decoded_preds, decoded_labels)

# After the epoch
cer = cer_metric.compute()
wer = wer_metric.compute()
print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, CER: {cer:.4f}, WER: {wer:.4f}")