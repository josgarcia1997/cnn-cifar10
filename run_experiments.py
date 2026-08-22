import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report

from src.utils import seed_everything, detectar_dispositivo
from src.model import CIFAR10CNN, CIFAR10CNN_Deep
from src.train import entrenar
from src.evaluate import predecir

def main():
    device = detectar_dispositivo()
    seed_everything(42)
    batch_size = 64
    max_epochs = 10
    
    transform_base = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    transform_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset_base = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_base)
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_base)
    train_dataset_aug = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_aug)
    
    train_loader_base = torch.utils.data.DataLoader(train_dataset_base, batch_size=batch_size, shuffle=True)
    train_loader_aug = torch.utils.data.DataLoader(train_dataset_aug, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    experiments = [
        ("CNN básica", CIFAR10CNN().to(device), train_loader_base),
        ("CNN profunda", CIFAR10CNN_Deep().to(device), train_loader_base),
        ("CNN + Dropout", CIFAR10CNN(dropout=0.5).to(device), train_loader_base),
        ("CNN + Data Augmentation", CIFAR10CNN().to(device), train_loader_aug),
    ]
    
    results = {}
    best_model_name = None
    best_model_acc = 0.0
    best_model_state = None
    best_history = None
    
    for name, model, loader in experiments:
        print(f"\\n--- Entrenando {name} ---")
        seed_everything(42)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model, historia = entrenar(model, loader, test_loader, criterion, optimizer, device, max_epochs=max_epochs, verbose=False)
        
        y_real, y_pred, _ = predecir(model, test_loader, device)
        report = classification_report(y_real, y_pred, output_dict=True)
        
        test_acc = report['accuracy']
        train_acc = historia.train_acc[-1]
        prec = report['macro avg']['precision']
        rec = report['macro avg']['recall']
        f1 = report['macro avg']['f1-score']
        
        results[name] = {
            'Train Acc': train_acc,
            'Test Acc': test_acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1
        }
        print(f"Finalizado {name}: Test Acc = {test_acc:.4f}")
        
        if test_acc > best_model_acc:
            best_model_acc = test_acc
            best_model_name = name
            import copy
            best_model_state = copy.deepcopy(model.state_dict())
            best_history = historia
            
    print(f"\\nMejor modelo: {best_model_name} (Acc: {best_model_acc:.4f})")
    
    # Save the best model
    os.makedirs('models', exist_ok=True)
    torch.save(best_model_state, "models/cifar10_cnn.pth")
    
    # Generate table
    table_str = "| Modelo | Train Accuracy | Test Accuracy | Precision | Recall | F1 |\\n|---|---:|---:|---:|---:|---:|\\n"
    for name, m in results.items():
        table_str += f"| {name} | {m['Train Acc']:.4f} | {m['Test Acc']:.4f} | {m['Precision']:.4f} | {m['Recall']:.4f} | {m['F1']:.4f} |\\n"
        
    with open('resultados/tabla_comparativa.md', 'w') as f:
        f.write(table_str)
        
    # Generate and save plots for best model
    os.makedirs('results', exist_ok=True)
    plt.figure()
    plt.plot(best_history.train_loss, label='Train Loss')
    plt.plot(best_history.val_loss, label='Val Loss')
    plt.title(f'Training Loss ({best_model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('results/training_loss.png')
    plt.close()
    
    plt.figure()
    plt.plot(best_history.train_acc, label='Train Acc')
    plt.plot(best_history.val_acc, label='Val Acc')
    plt.title(f'Training Accuracy ({best_model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('results/training_accuracy.png')
    plt.close()

if __name__ == '__main__':
    main()
