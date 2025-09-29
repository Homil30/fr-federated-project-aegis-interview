import torch.nn as nn
import torch.nn.functional as F

class SimpleFaceNet(nn.Module):
    def __init__(self, num_classes=2):  # Male / Female
        super(SimpleFaceNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [32, 112, 112]
        x = self.pool(F.relu(self.conv2(x)))   # [64, 56, 56]
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
