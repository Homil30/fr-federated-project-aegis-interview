"""
Complete Evaluation Suite
Runs all evaluations: Fairness, Adversarial Robustness, and generates final outputs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import json
from tqdm import tqdm

# Fix: JSON encoder for NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class FaceRecognitionModel(nn.Module):
    """Face Recognition Model - matches your trained model"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class SyntheticTestDataset(Dataset):
    """Dataset for synthetic test data"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []
        
        for gender_dir in self.data_dir.iterdir():
            if gender_dir.is_dir():
                gender = gender_dir.name
                for race_dir in gender_dir.iterdir():
                    if race_dir.is_dir():
                        race = race_dir.name
                        for img_path in race_dir.glob("*.jpg"):
                            label = 0 if gender == 'Male' else 1
                            demo_str = f"{gender}_{race}"
                            self.samples.append((img_path, label, demo_str))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, demo_str = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {"image": image, "label": label, "demographics": demo_str}

def collate_fn(batch):
    """Custom collate function"""
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    demographics = [item["demographics"] for item in batch]
    return {"image": images, "label": labels, "demographics": demographics}

def fgsm_attack(model, images, labels, epsilon=0.1):
    """FGSM adversarial attack"""
    images.requires_grad = True
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()
    attack_images = images + epsilon * images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)
    return attack_images

def evaluate_fairness(model, dataloader, device):
    """Evaluate model fairness across demographic groups"""
    print("\n" + "="*60)
    print("FAIRNESS EVALUATION")
    print("="*60)
    model.eval()
    group_metrics = {}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating fairness"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            demographics = batch['demographics']

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            for i in range(len(labels)):
                demo_str = demographics[i]

                if demo_str not in group_metrics:
                    group_metrics[demo_str] = {
                        'correct': 0,
                        'total': 0,
                        'true_positives': 0,
                        'false_positives': 0,
                        'true_negatives': 0,
                        'false_negatives': 0
                    }

                is_correct = (predicted[i] == labels[i]).item()
                group_metrics[demo_str]['total'] += 1
                if is_correct:
                    group_metrics[demo_str]['correct'] += 1

                pred = predicted[i].item()
                true = labels[i].item()

                if pred == 1 and true == 1:
                    group_metrics[demo_str]['true_positives'] += 1
                elif pred == 1 and true == 0:
                    group_metrics[demo_str]['false_positives'] += 1
                elif pred == 0 and true == 0:
                    group_metrics[demo_str]['true_negatives'] += 1
                elif pred == 0 and true == 1:
                    group_metrics[demo_str]['false_negatives'] += 1

    results = {}
    for group, m in group_metrics.items():
        acc = m['correct'] / m['total'] if m['total'] else 0
        tp, fp, fn = m['true_positives'], m['false_positives'], m['false_negatives']
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

        results[group] = {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'sample_count': int(m['total'])
        }

    return results

def evaluate_adversarial_robustness(model, dataloader, device, epsilons=[0.0, 0.05, 0.1, 0.15, 0.2]):
    """Evaluate model robustness against adversarial attacks"""
    print("\n" + "="*60)
    print("ADVERSARIAL ROBUSTNESS EVALUATION")
    print("="*60)
    
    results = {}
    
    for epsilon in epsilons:
        print(f"Testing epsilon = {epsilon}")
        model.eval()
        correct = 0
        total = 0
        
        for batch in tqdm(dataloader, desc=f"ε={epsilon}"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            if epsilon > 0:
                images = fgsm_attack(model, images, labels, epsilon)
            
            with torch.no_grad():
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        results[f"epsilon_{epsilon}"] = {
            'epsilon': float(epsilon),
            'accuracy': float(accuracy),
            'correct': int(correct),
            'total': int(total)
        }
        print(f"  Accuracy: {accuracy:.4f}")
    
    return results

def create_fairness_plots(fairness_results, output_dir="evaluation_results"):
    """Create visualizations for fairness metrics"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Prepare data for plotting
    groups = list(fairness_results.keys())
    accuracies = [fairness_results[g]['accuracy'] for g in groups]
    f1_scores = [fairness_results[g]['f1_score'] for g in groups]
    sample_counts = [fairness_results[g]['sample_count'] for g in groups]
    
    # Plot 1: Accuracy by demographic group
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(groups)), accuracies, color='steelblue')
    plt.xlabel('Demographic Group')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Across Demographic Groups')
    plt.xticks(range(len(groups)), groups, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fairness_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: F1 Score by demographic group
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(groups)), f1_scores, color='coral')
    plt.xlabel('Demographic Group')
    plt.ylabel('F1 Score')
    plt.title('Model F1 Score Across Demographic Groups')
    plt.xticks(range(len(groups)), groups, rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.grid(axis='y', alpha=0.3)
    
    for i, (bar, f1) in enumerate(zip(bars, f1_scores)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fairness_f1score.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Heatmap of all metrics
    metrics_data = []
    for group in groups:
        metrics_data.append([
            fairness_results[group]['accuracy'],
            fairness_results[group]['precision'],
            fairness_results[group]['recall'],
            fairness_results[group]['f1_score']
        ])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(metrics_data, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                yticklabels=groups, cbar_kws={'label': 'Score'})
    plt.title('Fairness Metrics Heatmap Across Demographic Groups')
    plt.tight_layout()
    plt.savefig(output_dir / 'fairness_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Fairness plots saved to {output_dir}/")

def create_robustness_plots(robustness_results, output_dir="evaluation_results"):
    """Create visualizations for adversarial robustness"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Extract data
    epsilons = [robustness_results[k]['epsilon'] for k in sorted(robustness_results.keys())]
    accuracies = [robustness_results[k]['accuracy'] for k in sorted(robustness_results.keys())]
    
    # Plot: Accuracy vs Epsilon
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, accuracies, marker='o', linewidth=2, markersize=8, color='darkred')
    plt.xlabel('Epsilon (Attack Strength)')
    plt.ylabel('Accuracy')
    plt.title('Model Robustness: Accuracy vs Adversarial Attack Strength')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # Add value labels
    for eps, acc in zip(epsilons, accuracies):
        plt.text(eps, acc + 0.02, f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'robustness_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Robustness plots saved to {output_dir}/")

def main():
    print("="*70)
    print(" COMPLETE EVALUATION SUITE ")
    print("="*70)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_path = Path("global_model.pt")
    test_data_dir = Path("synthetic_test_data")
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = FaceRecognitionModel(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("✓ Model loaded successfully")
    
    # Prepare data
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SyntheticTestDataset(test_data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    print(f"✓ Loaded {len(dataset)} test samples")
    
    # Run evaluations
    results = {}
    
    # 1. Fairness Evaluation
    fairness_results = evaluate_fairness(model, dataloader, device)
    results['fairness'] = fairness_results
    create_fairness_plots(fairness_results, output_dir)
    
    # 2. Adversarial Robustness Evaluation
    robustness_results = evaluate_adversarial_robustness(model, dataloader, device)
    results['adversarial_robustness'] = robustness_results
    create_robustness_plots(robustness_results, output_dir)
    
    # Save results to JSON - FIXED LINE HERE
    results_file = output_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70)
    print(f"✓ Results saved to: {results_file}")
    print(f"✓ Plots saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - evaluation_results.json")
    print("  - fairness_accuracy.png")
    print("  - fairness_f1score.png")
    print("  - fairness_heatmap.png")
    print("  - robustness_curve.png")

if __name__ == "__main__":
    main()