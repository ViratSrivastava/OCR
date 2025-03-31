import torch
from torchviz import make_dot
from torch import nn
import torchvision.models as models

class CRNN(nn.Module):
    def __init__(self, num_classes=37):  # 26 letters + 10 digits + blank
        super(CRNN, self).__init__()

        # CNN Feature Extractor (ResNet18 without the last FC layer)
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # Remove last pooling & FC

        # LSTM for Sequence Modeling
        self.rnn = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, 
                           bidirectional=True, batch_first=True)

        # Fully Connected Layer (Mapping to Character Classes)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # Extract Features from CNN
        x = self.cnn(x)  # Shape: (B, C, H, W)

        # Reshape for LSTM: (batch, time_steps, features)
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C) → (B, W, H, C)
        x = x.view(batch_size, x.size(1), -1)  # Flatten height and channels
        
        # LSTM Sequence Modeling
        x, _ = self.rnn(x)

        # Output Predictions
        x = self.fc(x)  # Shape: (B, W, num_classes)
        return x

# Example usage
num_classes = 128
model = CRNN(num_classes)
dummy_input = torch.randn(1, 1, 32, 128)  # Batch size 1, grayscale image (1 channel), height 32, width 128
output = model(dummy_input)
make_dot(output, params=dict(model.named_parameters())).render("CRNN", format="png")