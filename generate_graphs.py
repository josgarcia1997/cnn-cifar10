import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import os

from src.utils import seed_everything, detectar_dispositivo
from src.model import CIFAR10CNN
from src.evaluate import predecir, errores_alta_confianza

def imshow(img):
    img = img / 2 + 0.5  # desnormalizar
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def main():
    seed_everything(42)
    device = detectar_dispositivo()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    
    # Load model
    model = CIFAR10CNN().to(device)
    model.load_state_dict(torch.load("models/cifar10_cnn.pth", map_location=device))
    model.eval()
    
    print("Prediciendo...")
    y_real, y_pred, confianzas = predecir(model, test_loader, device)
    
    print("Generando matriz de confusión...")
    cm = confusion_matrix(y_real, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    
    print("Generando predicciones...")
    correct_idx = [i for i, (r, p) in enumerate(zip(y_real, y_pred)) if r == p][:5]
    incorrect_idx = [i for i, (r, p) in enumerate(zip(y_real, y_pred)) if r != p][:5]
    
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(correct_idx + incorrect_idx):
        img, _ = test_dataset[idx]
        plt.subplot(2, 5, i+1)
        imshow(img)
        color = 'green' if i < 5 else 'red'
        plt.title(f"Real: {class_names[y_real[idx]]}\\nPred: {class_names[y_pred[idx]]}", color=color)
        plt.axis('off')
    plt.savefig('results/predictions.png')
    plt.close()
    
    print("Generando feature maps...")
    img, _ = test_dataset[0]
    img_input = img.unsqueeze(0).to(device)
    with torch.inference_mode():
        features = model.features[0](img_input) # Primer Conv2d
        features = F.relu(features)
    
    features = features.squeeze(0).cpu().numpy()
    plt.figure(figsize=(12, 12))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(features[i], cmap='viridis')
        plt.axis('off')
    plt.suptitle('Primeros 16 Feature Maps (Capa 1)')
    plt.savefig('results/feature_maps.png')
    plt.close()
    
    print("Terminado.")

if __name__ == '__main__':
    main()
