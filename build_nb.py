import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("# Parte 1 — Exploración del dataset"))
cells.append(nbf.v4.new_code_cell("""import sys
sys.path.append('..')
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

from src.utils import seed_everything, detectar_dispositivo

seed_everything(42)
device = detectar_dispositivo()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root="../data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root="../data", train=False, download=True, transform=transform)

batch_size = 64
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
"""))

cells.append(nbf.v4.new_markdown_cell("## Inspeccionar dimensiones"))
cells.append(nbf.v4.new_code_cell("""img, label = train_dataset[0]
print(f"Shape del tensor: {img.shape}")
print(f"Label numérico: {label}")
print(f"Nombre de la clase: {class_names[label]}")
"""))

cells.append(nbf.v4.new_markdown_cell("## Visualizar imágenes"))
cells.append(nbf.v4.new_code_cell("""def imshow(img):
    img = img / 2 + 0.5  # desnormalizar
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(train_loader)
images, labels = next(dataiter)

plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i+1)
    imshow(images[i])
    plt.title(class_names[labels[i]])
    plt.axis('off')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""## Analizar las clases
1. **¿Cuántas clases existen?** 10 clases.
2. **¿Cuántas imágenes tiene training?** 50,000 imágenes.
3. **¿Cuántas imágenes tiene test?** 10,000 imágenes.
4. **¿El dataset está balanceado?** Sí, 5,000 imágenes por clase en train.
5. **¿Qué clases parecen más fáciles de distinguir?** Vehículos como Airplane, Automobile y Ship vs animales.
6. **¿Qué clases podrían confundirse?** Cat vs Dog, y Automobile vs Truck."""))

cells.append(nbf.v4.new_markdown_cell("# Parte 2 — Comprender las convoluciones"))
cells.append(nbf.v4.new_code_cell("""import torch.nn as nn
import torch.nn.functional as F

conv = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
out = conv(img.unsqueeze(0))
print(f"Dimensión de entrada: {img.unsqueeze(0).shape}")
print(f"Dimensión de salida: {out.shape}")
"""))

cells.append(nbf.v4.new_markdown_cell("""## Respuestas a convoluciones
1. **¿Qué significa in_channels=3?** La imagen tiene 3 canales (RGB).
2. **¿Qué significa out_channels=16?** La capa generará 16 feature maps distintos (usará 16 filtros/kernels).
3. **¿Qué representa kernel_size=3?** El tamaño espacial de cada filtro es de 3x3 píxeles.
4. **¿Para qué sirve padding=1?** Para añadir un borde de ceros alrededor de la imagen, permitiendo que la salida mantenga el mismo tamaño espacial que la entrada (32x32) tras una convolución 3x3.
5. **¿Por qué se obtienen 16 feature maps?** Porque hemos especificado 16 filtros distintos en `out_channels`, cada uno aprendiendo a extraer una característica visual diferente."""))

cells.append(nbf.v4.new_markdown_cell("# Parte 3 — Construcción de la CNN"))
cells.append(nbf.v4.new_code_cell("""from src.model import CIFAR10CNN
model = CIFAR10CNN().to(device)
print(model)
"""))

cells.append(nbf.v4.new_markdown_cell("""# Parte 4 — Analizar dimensiones
1. **¿Por qué disminuye el tamaño espacial?** Por la operación de `MaxPool2d`, que en este caso reduce las dimensiones espaciales a la mitad en cada paso al tomar el valor máximo de cada cuadrante 2x2.
2. **¿Por qué aumenta el número de canales?** Porque en cada capa convolucional incrementamos `out_channels` para permitir a la red extraer características cada vez más complejas y abstractas a costa de resolución espacial.
3. **¿Qué información intenta aprender cada filtro?** Los primeros filtros aprenden características de bajo nivel (bordes, colores, texturas), mientras que los más profundos aprenden combinaciones de estas características (ojos, ruedas, formas complejas).
4. **¿Qué función cumple Flatten?** Aplanar los feature maps (convertir un tensor 3D de $128 \\times 4 \\times 4$ a un vector 1D de 2048 elementos) para que pueda ser procesado por las capas densas (`Linear`) de clasificación."""))

cells.append(nbf.v4.new_markdown_cell("# Parte 5, 6 y 7 — Entrenamiento y Visualización"))
cells.append(nbf.v4.new_code_cell("""import torch.optim as optim
from src.train import entrenar

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Usamos pocas epochs (ej. 5) en el notebook inicial para demostrar funcionamiento rápido
model, historia = entrenar(model, train_loader, test_loader, criterion, optimizer, device, max_epochs=5)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(historia.train_loss, label='Train')
plt.plot(historia.val_loss, label='Test/Val')
plt.title('Training Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../results/training_loss.png')

plt.subplot(1,2,2)
plt.plot(historia.train_acc, label='Train')
plt.plot(historia.val_acc, label='Test/Val')
plt.title('Training Accuracy vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('../results/training_accuracy.png')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""## Análisis de Entrenamiento
1. **¿Disminuye el loss?** Sí, indicando que el modelo ajusta sus pesos.
2. **¿Aumenta el accuracy?** Sí, en entrenamiento sube sostenidamente.
3. **¿Parece que el modelo aprende?** Sí, aunque en Test el progreso puede frenarse si hay overfitting.
4. **¿Hay señales de inestabilidad?** Dependerá de las curvas, pero sin regularización es común que Test empiece a divergir lentamente."""))

cells.append(nbf.v4.new_markdown_cell("# Parte 8 y 9 — Evaluación y Matriz de Confusión"))
cells.append(nbf.v4.new_code_cell("""from src.evaluate import predecir
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

y_real, y_pred, confianzas = predecir(model, test_loader, device)

print("Classification Report:")
print(classification_report(y_real, y_pred, target_names=class_names))

cm = confusion_matrix(y_real, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Matriz de Confusión")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.savefig('../results/confusion_matrix.png')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""## Análisis de Matriz de Confusión
1. **¿Qué clase se identifica mejor?** Generalmente Ship, Automobile o Frog (varía por seed).
2. **¿Qué clase se identifica peor?** Bird, Cat o Dog.
3. **¿Qué clases se confunden?** Cat con Dog, y Automobile con Truck.
4. **¿Por qué podrían confundirse?** Cat y Dog comparten muchas características visuales y colores similares. Automobile y Truck también son vehículos de 4 ruedas con geometrías parecidas en baja resolución."""))

cells.append(nbf.v4.new_markdown_cell("# Parte 10 — Visualización de predicciones"))
cells.append(nbf.v4.new_code_cell("""correct_idx = [i for i, (r, p) in enumerate(zip(y_real, y_pred)) if r == p][:5]
incorrect_idx = [i for i, (r, p) in enumerate(zip(y_real, y_pred)) if r != p][:5]

plt.figure(figsize=(15, 6))
for i, idx in enumerate(correct_idx + incorrect_idx):
    img, _ = test_dataset[idx]
    plt.subplot(2, 5, i+1)
    imshow(img)
    color = 'green' if i < 5 else 'red'
    plt.title(f"Real: {class_names[y_real[idx]]}\\nPred: {class_names[y_pred[idx]]}", color=color)
    plt.axis('off')
plt.savefig('../results/predictions.png')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("# Parte 11 — Análisis de errores"))
cells.append(nbf.v4.new_code_cell("""from src.evaluate import errores_alta_confianza
errores = errores_alta_confianza(y_real, y_pred, confianzas, top_k=10)

plt.figure(figsize=(20, 8))
for i, (idx, real, pred, conf) in enumerate(errores):
    img, _ = test_dataset[idx]
    plt.subplot(2, 5, i+1)
    imshow(img)
    plt.title(f"Real: {class_names[real]}\\nPred: {class_names[pred]}\\nConf: {conf:.2f}", color='red', fontsize=10)
    plt.axis('off')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""## Análisis de errores de alta confianza
1. **¿La imagen es ambigua?** En muchos casos (e.g., gatos vistos de lejos o aves raras).
2. **¿El objeto ocupa poco espacio?** Sí, a 32x32, el objeto principal a veces es solo unos píxeles.
3. **¿El fondo puede estar confundiendo la CNN?** Definitivamente. Si un perro está sobre el agua, podría inferirse 'Ship' o 'Frog'.
4. **¿La baja resolución puede explicar el error?** Sí, 32x32 carece del detalle fino para distinguir textura de pelaje.
5. **¿La clase predicha tiene similitudes con la correcta?** Sí, confundir un ciervo con un caballo es común por ser cuadrúpedos de colores similares."""))

cells.append(nbf.v4.new_markdown_cell("# Parte 12 — Visualización de Feature Maps"))
cells.append(nbf.v4.new_code_cell("""img, _ = test_dataset[0]
img_input = img.unsqueeze(0).to(device)
model.eval()

# Extraer feature maps de la primera capa
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
plt.savefig('../results/feature_maps.png')
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""## Explicación Feature Maps
- **Por qué los mapas son diferentes:** Porque cada filtro de convolución aprende parámetros (pesos) únicos durante el entrenamiento, buscando patrones distintos.
- **Qué pueden estar detectando:** Algunos detectan bordes verticales, otros horizontales, otros contraste de colores o texturas específicas de la imagen.
- **Por qué no todos responden igual:** Se debe a la inicialización aleatoria de pesos y cómo el gradiente empuja a cada filtro a especializarse en diferentes características para minimizar el Loss global."""))

nb['cells'] = cells
with open('c:/Users/USUARIO/Desktop/MAESTRIA EN CIENCIAS DE DATOS/DeepLearning/Taller_Final/cnn-cifar10/notebooks/01_cifar10_exploracion_entrenamiento.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
