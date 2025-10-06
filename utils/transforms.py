from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((160, 160)),  # Resize to FaceNet input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])
