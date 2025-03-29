import torch
import torch.nn as nn
import torch.nn.functional as F

class OCRNet(nn.Module):
    def __init__(self, num_classes):
        super(OCRNet, self).__init__()

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(512, 512, kernel_size=2),
            nn.ReLU()
        )

        # Recurrent Layer (LSTM)
        self.lstm = nn.LSTM(512, 256, bidirectional=True)

        # Fully Connected Layer (Mapping to character classes)
        self.fc = nn.Linear(512 * 2, num_classes)  # 512 * 2 for bidirectional LSTM

    def forward(self, x):
        print(f"Input shape to CNN: {x.shape}")
        x = self.cnn(x)
        print(f"Shape after CNN: {x.shape}")
        
        x = x.squeeze(2)  
        print(f"Shape after squeeze: {x.shape}")
        
        x = x.permute(0, 2, 1)  
        print(f"Shape after permute: {x.shape}")
        
        x_lstm_outs,_ = self.lstm(x)
        print(f"Shape after LSTM: {x_lstm_outs.shape}")
        
        x = self.fc(x_lstm_outs)
        print(f"Shape before softmax: {x.shape}")
        
        return F.log_softmax(x, dim=2)

