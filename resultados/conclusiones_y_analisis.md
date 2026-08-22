# Conclusiones y Análisis de Experimentos (CIFAR-10)

## Resumen de la experimentación
Como parte de la Parte 13 del Taller Final, se ha llevado a cabo un estudio comparativo utilizando 4 variantes de arquitecturas de Redes Neuronales Convolucionales (CNN):
1. **CNN Básica**: Una red de 3 capas convolucionales profundizando los filtros (32 -> 64 -> 128) con Max Pooling.
2. **CNN Profunda**: Una red con bloques de dos capas convolucionales consecutivas antes de cada Max Pooling, permitiendo aprender características más abstractas (mayor *Receptive Field* efectivo).
3. **CNN + Dropout**: La red básica a la cual se le añadió regularización Dropout (p=0.5) en la capa fully connected para mitigar el sobreajuste.
4. **CNN + Data Augmentation**: La red básica entrenada con un dataset aumentado (RandomHorizontalFlip y RandomCrop) para fomentar invarianza geométrica.

## Tabla Comparativa de Rendimiento

| Modelo | Train Accuracy | Test Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| CNN básica | 0.9185 | 0.7415 | 0.7472 | 0.7415 | 0.7413 |
| CNN profunda | 0.9192 | 0.7738 | 0.7773 | 0.7738 | 0.7742 |
| CNN + Dropout | 0.8403 | 0.7644 | 0.7669 | 0.7644 | 0.7648 |
| CNN + Data Augmentation | 0.7779 | 0.7841 | 0.7914 | 0.7841 | 0.7838 |

## Análisis
- **El Impacto de la Profundidad:** Al añadir más capas (CNN Profunda), la red tiene mayor capacidad de representación, pero requiere más tiempo para converger y puede ser más propensa a sobreajuste si el dataset no es lo suficientemente masivo o si no se utiliza regularización exhaustiva.
- **El Papel del Dropout:** Evita que el modelo dependa en exceso de ciertas características "fáciles" del set de entrenamiento, forzándolo a aprender patrones más robustos. En ocasiones ralentiza la convergencia inicial, pero resulta en mejor generalización (menor brecha entre el accuracy de entrenamiento y validación).
- **Eficacia del Data Augmentation:** Es la técnica de regularización más robusta para imágenes, ya que simula variaciones naturales (rotación, perspectiva, escala), obligando al modelo a aprender el concepto del objeto independientemente de su posición, reduciendo notablemente el overfitting.
- **Clases difíciles:** Aún con las mejoras, el problema del bajo detalle (imágenes de 32x32) hace que categorías semánticamente cercanas en contornos y colores (Perro vs Gato, Automóvil vs Camión) sigan teniendo un porcentaje significativo de error, especialmente cuando los sujetos se mezclan con fondos complejos.

## Conclusión Central
El modelo logra clasificar imágenes aprendiendo características de lo general a lo particular. La optimización mediante *Backpropagation* sobre los mapas de características permite a los kernels convolucionales actuar como extractores automáticos que ninguna regla programada tradicionalmente podría igualar. Experimentos de regularización como Dropout y Data Augmentation confirman que "memorizar" el entrenamiento no es útil, y que forzar al modelo a extraer patrones esenciales es el secreto de la verdadera Inteligencia Artificial aplicada a Visión.
