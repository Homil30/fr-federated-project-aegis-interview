# client.py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import flwr as fl

from model import SimpleFaceNet
from utils.aggregation_utils import parameters_to_state_dict


# -------------------------------
# Training & Evaluation
# -------------------------------
def train(model, trainloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(trainloader)


def test(model, testloader, criterion, device):
    model.eval()
    correct, total, loss_total = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss_total += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total if total > 0 else 0
    return loss_total / len(testloader), accuracy


# -------------------------------
# Flower Client
# -------------------------------
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, trainloader, valloader, device):
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = parameters_to_state_dict(parameters, self.model.state_dict())
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model, self.trainloader, self.criterion, self.optimizer, self.device)
        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valloader, self.criterion, self.device)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8081")
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"📂 Loading dataset from {args.data_dir} ...")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # ✅ Load the full dataset (no train/val folders needed)
    full_dataset = datasets.ImageFolder(root=args.data_dir, transform=transform)

    # ✅ Split 80/20 into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"✓ Dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val samples.")

    # ✅ Initialize model
    model = SimpleFaceNet(num_classes=len(full_dataset.classes)).to(device)

    # ✅ Start Flower client
    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient(model, trainloader, valloader, device).to_client(),
    )


if __name__ == "__main__":
    main()
