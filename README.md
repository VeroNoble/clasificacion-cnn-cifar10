# Clasificación de imágenes con CNN (CIFAR-10)

Trabajo práctico de la Tecnicatura Superior en Ciencia de Datos e IA.

## Objetivo

Entrenar una red neuronal convolucional (CNN) desde cero para clasificar imágenes en 10 categorías (avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco, camión), usando el dataset CIFAR-10.

## Datos

Dataset **CIFAR-10** (60.000 imágenes de 32x32 píxeles a color, 10 clases), cargado directamente desde `keras.datasets.cifar10`.

## Enfoque

1. **Data Augmentation**: flips horizontales y rotaciones aleatorias para reducir el sobreajuste.
2. **Arquitectura**: CNN de 3 bloques convolucionales (32 → 64 → 64 filtros) con Batch Normalization y Max Pooling, seguida de una capa densa de 128 unidades con Dropout (0.4) y una capa de salida de 10 clases.
3. **Entrenamiento**: optimizador Adam, hasta 40 épocas, con `EarlyStopping` (detiene si no mejora en 8 épocas), `ModelCheckpoint` (guarda solo el mejor modelo) y `ReduceLROnPlateau` (reduce la tasa de aprendizaje si el error se estanca).
4. **Prueba interactiva**: un script aparte (`probar_modelo.py`) permite elegir cualquier imagen desde el explorador de archivos y ver la predicción del modelo en pantalla, recortada y redimensionada automáticamente a 32x32.

## Resultados

El modelo alcanza un **accuracy de 80,66%** sobre las 10.000 imágenes de test de CIFAR-10 (10 clases; el azar daría 10%). Como el conjunto de test también se utilizó como validación durante el entrenamiento (para `EarlyStopping` y `ModelCheckpoint`), esta cifra puede ser levemente optimista; una evaluación más estricta separaría un conjunto de validación propio a partir de los datos de entrenamiento.

El modelo entrenado se incluye en el repositorio (`mejor_modelo.keras`), guardado automáticamente en su mejor punto de validación gracias a `ModelCheckpoint`, por lo que puede probarse directamente con `probar_modelo.py` sin volver a entrenar.

## Herramientas

Python · TensorFlow / Keras · NumPy · Matplotlib · Pillow

## Cómo correrlo

```bash
pip install -r requirements.txt

# Para reentrenar el modelo desde cero:
python entrenar.py

# Para probar el modelo ya entrenado con tus propias imágenes:
python probar_modelo.py
```

`probar_modelo.py` abre una ventana para elegir una imagen desde tu computadora y muestra la predicción del modelo con su porcentaje de confianza.
