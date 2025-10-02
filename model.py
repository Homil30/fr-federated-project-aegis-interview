# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFaceNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleFaceNet, self).__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112x112

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56x56

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 14x14
        )

        # ✅ Fix: use adaptive pooling → always outputs 7x7
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),  # 12544 → 512
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)  # fixed output size
        x = self.classifier(x)
        return x


# ✅ Quick test (optional)
if __name__ == "__main__":
    model = SimpleFaceNet(num_classes=2)
    dummy = torch.randn(4, 3, 224, 224)  # batch of 4 images
    out = model(dummy)
    print("Output shape:", out.shape)  # should be [4, 2]
