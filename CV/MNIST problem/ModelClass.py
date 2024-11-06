import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self, in_channels = 1, num_classes = 10, hidden = 384):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        
        
        self.seq1 = nn.Sequential(
			nn.Conv2d(in_channels, 6, 3, padding=1), # 28 -> 28
            self.relu,
            nn.BatchNorm2d(6),
			nn.MaxPool2d(2, 2) # 28 -> 14
		)
        
        self.seq2 = nn.Sequential(
			nn.Conv2d(6, 16, 3, padding=1), # 14 -> 14
            self.relu,
            nn.BatchNorm2d(16),
			nn.MaxPool2d(2, 2) # 14 -> 7
		)
        self.fc1 = nn.Linear(7 * 7 * 16, hidden)
        self.bn = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
    
    def forward(self, x):
        x = x.unsqueeze(dim = 1)
        x = self.seq1(x)
        x = self.seq2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.bn(x)
        x = self.fc2(x)
        return x