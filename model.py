import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights

try:
    from facenet_pytorch import InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False


class SimpleFaceNet(nn.Module):
    def __init__(self, embedding_size=128, pretrained=True, backbone="resnet18"):
        super().__init__()

        if backbone == "resnet18":
            # ✅ Option A: ResNet18 backbone (ImageNet pretrained)
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            base.fc = nn.Identity()
            self.backbone = base
            feat_dim = 512

        elif backbone == "facenet" and FACENET_AVAILABLE:
            # ✅ Option B: FaceNet backbone (VGGFace2 pretrained)
            self.backbone = InceptionResnetV1(pretrained="vggface2")
            feat_dim = 512  # InceptionResnetV1 outputs 512-dim embeddings

        else:
            raise ValueError("Unsupported backbone. Use 'resnet18' or 'facenet'.")

        # Projection to desired embedding size
        self.fc = nn.Linear(feat_dim, embedding_size)

    def forward(self, x):
        feat = self.backbone(x)
        if feat.dim() > 2:  # flatten if needed
            feat = torch.flatten(feat, 1)
        emb = self.fc(feat)
        emb = F.normalize(emb, p=2, dim=1)  # L2-normalize embeddings
        return emb
