# Clasificación de imágenes con Redes Neuronales Convolucionales (CNN) usando CIFAR-10

## 1. Problema
El objetivo de este proyecto es construir un sistema de visión artificial capaz de clasificar imágenes automáticamente en 10 categorías distintas, demostrando la eficacia de las Redes Neuronales Convolucionales (CNN) en comparación con las redes densas tradicionales, mediante el aprendizaje iterativo de representaciones visuales complejas a partir del dataset CIFAR-10.

## 2. Dataset
Se utiliza el dataset público **CIFAR-10**, compuesto por 60,000 imágenes a color de 32x32 píxeles, organizadas equitativamente en 10 clases (Avión, Automóvil, Pájaro, Gato, Ciervo, Perro, Rana, Caballo, Barco y Camión). El dataset está balanceado e incluye 50,000 imágenes para entrenamiento y 10,000 para evaluación.

## 3. Arquitectura
La red principal (`CIFAR10CNN`) sigue el patrón secuencial estándar de extracción de características seguido de clasificación:
- 3 bloques convolucionales: `Conv2d -> ReLU -> MaxPool2d`. (Los filtros aumentan progresivamente: 32 -> 64 -> 128).
- Aplanado de tensores (`Flatten`).
- Clasificador denso: `Linear (2048 a 256) -> ReLU -> [Dropout (en experimento)] -> Linear (256 a 10)`.

## 4. Entrenamiento
El modelo se entrenó optimizando la función de pérdida `CrossEntropyLoss` mediante el algoritmo `Adam` con un learning rate de 0.001 a lo largo de 10 épocas, utilizando un batch_size de 64.

## 5. Resultados y Gráficos
| Modelo | Train Accuracy | Test Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| CNN básica | 0.9185 | 0.7415 | 0.7472 | 0.7415 | 0.7413 |
| CNN profunda | 0.9192 | 0.7738 | 0.7773 | 0.7738 | 0.7742 |
| CNN + Dropout | 0.8403 | 0.7644 | 0.7669 | 0.7644 | 0.7648 |
| CNN + Data Augmentation | 0.7779 | 0.7841 | 0.7914 | 0.7841 | 0.7838 |

Como se observa, el uso de Data Augmentation logró el mejor resultado general, reduciendo enormemente el sobreajuste al disminuir el Train Accuracy a cambio del mejor Test Accuracy (78.41%). Los gráficos detallados del mejor modelo pueden encontrarse en la carpeta `results/`.

## 6. Análisis de Errores
Se detecta consistentemente confusión entre clases visualmente similares (ej. Gato vs Perro o Automóvil vs Camión), especialmente debido a que una resolución de 32x32 píxeles causa pérdida de características delgadas y finas de los animales, sumado a los ruidos de fondo en ciertas fotografías que desorientan al clasificador.

## 7. Preguntas Conceptuales
**¿Por qué una CNN es más adecuada que una ANN tradicional para imágenes?**
Las CNN mantienen y explotan la estructura espacial (2D) de las imágenes mediante kernels compartidos. Una ANN aplanaría la imagen desde el principio, perdiendo la relación de vecindad de píxeles y requiriendo muchos más parámetros (propensión a overfitting).

**¿Qué es una convolución?**
Una operación matemática lineal donde un filtro (kernel) se desliza por encima de la imagen multiplicando y sumando valores, para resaltar ciertas características locales.

**¿Qué es un kernel?**
Es una pequeña matriz de pesos que se desliza sobre la entrada durante la convolución. Actúa como un detector de patrones.

**¿Qué aprende un filtro?**
A lo largo del entrenamiento, aprende parámetros que maximizan su respuesta ante características visuales útiles (ej. primeros filtros aprenden bordes, capas más profundas aprenden partes de objetos como "ojos" o "ruedas").

**¿Qué es un feature map?**
Es el resultado de pasar un filtro convolucional sobre una entrada. Cada mapa de características representa la "activación" o presencia de la característica detectada por ese filtro en distintas regiones espaciales.

**¿Qué función cumple ReLU?**
Introduce no linealidad en la red, desactivando valores negativos. Sin ReLU (u otra activación), toda la red sería una simple transformación lineal, incapaz de modelar fronteras complejas.

**¿Qué función cumple MaxPooling?**
Reduce la dimensionalidad espacial (resolución) de los feature maps, disminuyendo el costo computacional y otorgando cierta invariancia a pequeñas traslaciones del objeto.

**¿Qué significa receptive field?**
Es la región (área de píxeles) de la imagen original de entrada que influye en la activación de un nodo específico en una capa profunda de la red.

**¿Qué función cumple el loss?**
Cuantifica la diferencia entre las predicciones del modelo y las etiquetas reales. Sirve como la señal o métrica de error que la red debe minimizar.

**¿Qué hace backpropagation?**
Calcula el gradiente (derivada) de la función de pérdida con respecto a todos los pesos de la red, enviando el error desde la salida hacia atrás hasta las primeras capas.

**¿Qué hace Adam?**
Es el algoritmo de optimización que actualiza los pesos de la red basándose en los gradientes calculados, adaptando dinámicamente la tasa de aprendizaje para cada parámetro.

**¿Qué representa el learning rate?**
El tamaño del "paso" que toma el optimizador en dirección contraria al gradiente. Si es muy grande puede divergir; si es muy bajo tardará muchísimo en converger.

**¿Qué significa epoch?**
Un ciclo completo en el que la red neuronal ha procesado el dataset de entrenamiento en su totalidad (una sola vez).

**¿Qué significa batch size?**
La cantidad de muestras procesadas juntas antes de actualizar los pesos de la red.

**¿Qué es overfitting?**
Cuando el modelo memoriza los datos de entrenamiento (logrando error bajo ahí) en lugar de aprender el patrón general, resultando en un mal desempeño frente a datos nuevos (test).

**¿Cómo puede detectarse?**
Cuando la pérdida (loss) en entrenamiento sigue bajando pero la pérdida en validación/test comienza a estancarse o subir.

**¿Cómo puede ayudar Dropout?**
Apaga aleatoriamente un porcentaje de neuronas en cada pasada de entrenamiento, forzando a la red a no depender de ninguna neurona en específico y distribuyendo el aprendizaje.

**¿Cómo puede ayudar Data Augmentation?**
Modifica aleatoriamente las imágenes (rotación, zoom, cortes) en cada época para generar variabilidad artificial, impidiendo que el modelo memorice imágenes exactas.

## 8. Conclusión (Pregunta Central)
**¿Cómo consigue aprender representaciones visuales útiles únicamente a partir de imágenes y etiquetas?**
Todo el proceso es guiado por un ciclo cerrado de retroalimentación matemática (Backpropagation). La imagen pasa por las convoluciones generando Feature Maps abstractos que el clasificador convierte en una predicción. Al evaluar esa predicción contra la etiqueta real, se genera un Loss (error). La clave radica en el cálculo de Gradientes, los cuales nos dicen exactamente en qué dirección y magnitud debemos modificar cada elemento de los Kernels para que el error sea menor la próxima vez. A través de miles de repeticiones de este proceso de actualización de kernels, los filtros comienzan a especializarse, casi "mágicamente", en características visuales (desde texturas hasta objetos) logrando representaciones robustas y útiles.
