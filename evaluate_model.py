import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.fairface_dataset import FairFaceDataset
from models.simple_facenet import SimpleFaceNet
import os

def evaluate_model(model_path, data_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleFaceNet().to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    val_csv = os.path.join(data_dir, "fairface-label-val.csv")
    img_dir = os.path.join(data_dir, "fairface-img-align")

    val_dataset = FairFaceDataset(val_csv, img_dir, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    print(f"✅ Model {model_path} Accuracy: {correct/total:.4f}")

if __name__ == "__main__":
    evaluate_model("saved_models/round-3.pt", "C:/Users/Asus/Desktop/XAI_Facial_Recognition/data/raw/FairFace")
