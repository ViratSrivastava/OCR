import torch
import torch.nn as nn
import torch.nn.functional as F

class OCRNet(nn.Module):
    def __init__(self, num_classes):
        super(OCRNet, self).__init__()
        
        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            # Initial: [B, 1, 32, 128]
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # -> [B, 64, 32, 128]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> [B, 64, 16, 64]
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # -> [B, 128, 16, 64]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> [B, 128, 8, 32]
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # -> [B, 256, 8, 32]
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # -> [B, 256, 8, 32]
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # -> [B, 256, 4, 32]
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # -> [B, 512, 4, 32]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # -> [B, 512, 4, 32]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # -> [B, 512, 2, 32]
            
            nn.Conv2d(512, 512, kernel_size=(2, 1), stride=1),  # -> [B, 512, 1, 32]
            nn.ReLU()
        )
        
        # Recurrent Layer (LSTM)
        self.lstm = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        
        # Fully Connected Layer
        self.fc = nn.Linear(512, num_classes)  # 512 because bidirectional LSTM outputs 256*2

    def forward(self, x):
        # CNN Feature Extraction
        x = self.cnn(x)  # -> [B, 512, 1, 32]
        
        # Prepare for LSTM
        x = x.squeeze(2)  # Remove height dimension -> [B, 512, 32]
        x = x.permute(0, 2, 1)  # -> [B, 32, 512]
        
        # LSTM sequence processing
        x, _ = self.lstm(x)  # -> [B, 32, 512]
        
        # Classification
        x = self.fc(x)  # -> [B, 32, num_classes]
        
        return x  # Return logits, apply log_softmax in loss calculation