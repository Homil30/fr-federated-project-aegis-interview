import flwr as fl
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from models.simple_facenet import SimpleFaceNet


# ------------------ Training + Evaluation ------------------ #
def train_one_epoch(model, loader, device):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()


def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss_sum += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return loss_sum / len(loader), correct / total


# ------------------ Flower Client ------------------ #
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(),
                              [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_one_epoch(self.model, self.train_loader, self.device)
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = evaluate(self.model, self.val_loader, self.device)
        return float(loss), len(self.val_loader.dataset), {"accuracy": float(acc)}


# ------------------ Main ------------------ #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8081")
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleFaceNet().to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # ✅ Load data directly from train/ and val/ folders
    train_dataset = datasets.ImageFolder(
        root=os.path.join(args.data_dir, "train"),
        transform=transform
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(args.data_dir, "val"),
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient(model, train_loader, val_loader, device).to_client(),
    )


if __name__ == "__main__":
    main()
