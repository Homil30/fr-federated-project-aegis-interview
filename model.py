import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFaceNet(nn.Module):
    def __init__(self, embedding_size=128):
        super(SimpleFaceNet, self).__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # ⚡ Instead of classifier → embedding generator
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, embedding_size),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.embedding(x)
        # Normalize embeddings (important for cosine similarity)
        x = F.normalize(x, p=2, dim=1)
        return x


# ✅ Test
if __name__ == "__main__":
    model = SimpleFaceNet(embedding_size=128)
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print("Embedding shape:", out.shape)  # [4, 128]
