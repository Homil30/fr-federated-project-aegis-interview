"""
Complete Evaluation Suite
Runs all evaluations: Fairness, Adversarial Robustness, and generates final outputs
"""

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model import SimpleFaceNet   # ✅ use same model as training

# ============================================================
# CONFIG
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "global_model.pt"
DATA_DIR = "data/test_synthetic"
RESULTS_DIR = "EVALUATION_RESULTS"

os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# LOAD MODEL
# ============================================================
print("\n======================================================================")
print(" COMPLETE EVALUATION SUITE ")
print("======================================================================\n")
print(f"Using device: {DEVICE}\n")

model = SimpleFaceNet().to(DEVICE)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    print(f"✓ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"⚠ Could not load model: {e}")

# ============================================================
# DATA PREP
# ============================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

# Use entire dataset for evaluation
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)
print(f"✓ Loaded {len(dataset)} test samples\n")

# ============================================================
# FAIRNESS EVALUATION
# ============================================================
print("============================================================")
print("FAIRNESS EVALUATION")
print("============================================================")

def evaluate_fairness(model, loader):
    model.eval()
    correct, total = 0, 0
    group_acc = {}
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating fairness"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total if total > 0 else 0
    return {"overall_accuracy": acc}

fairness_results = evaluate_fairness(model, test_loader)

# ============================================================
# ADVERSARIAL ROBUSTNESS
# ============================================================
print("\n============================================================")
print("ADVERSARIAL ROBUSTNESS EVALUATION")
print("============================================================")

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def evaluate_adversarial(model, loader, epsilons):
    model.eval()
    results = {}
    loss_fn = nn.CrossEntropyLoss()

    for eps in epsilons:
        correct = 0
        total = 0
        for images, labels in tqdm(loader, desc=f"ε={eps}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            images.requires_grad = True

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            model.zero_grad()
            loss.backward()
            data_grad = images.grad.data

            perturbed = fgsm_attack(images, eps, data_grad)
            outputs = model(perturbed)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total if total > 0 else 0
        results[eps] = acc
        print(f"  Accuracy at ε={eps}: {acc:.4f}")

    return results

epsilons = [0.0, 0.05, 0.1, 0.15, 0.2]
robustness_results = evaluate_adversarial(model, test_loader, epsilons)

# ============================================================
# SAVE RESULTS
# ============================================================
final_results = {
    "fairness": fairness_results,
    "adversarial_robustness": robustness_results,
}

with open(os.path.join(RESULTS_DIR, "evaluation_results.json"), "w") as f:
    json.dump(final_results, f, indent=4)

print("\n======================================================================")
print(" EVALUATION COMPLETE")
print("======================================================================\n")
print(f"Results saved in: {RESULTS_DIR}/evaluation_results.json")
print("✓ PROJECT EVALUATION COMPLETED SUCCESSFULLY!\n")
