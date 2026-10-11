import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Oculta los mensajes informativos de TensorFlow

# =====================================================================
# 1. CARGA Y NORMALIZACIÓN DE DATOS (estructura base provista por la cátedra)
# =====================================================================
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

(imagenes_entrenamiento, etiquetas_entrenamiento ), (imagenes_verificacion, etiquetas_verificacion) = keras.datasets.cifar10.load_data()

print(etiquetas_entrenamiento.shape, imagenes_entrenamiento.shape)
imagenes_entrenamiento, imagenes_verificacion = imagenes_entrenamiento / 255.0, imagenes_verificacion  / 255.0

nombres_clases = ['airplane','automobile','bird','cat','deer',
                  'dog','frog','horse','ship','truck']



# =====================================================================
# ARQUITECTURA Y ENTRENAMIENTO DEL MODELO
# =====================================================================

print("Construyendo la arquitectura...")

# 1. Aumento de datos: reduce el sobreajuste
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"), # Espejado horizontal aleatorio
    layers.RandomRotation(0.1),      # Rotación aleatoria de hasta ±10% de una vuelta
])

# 2. Construcción de la Red Neuronal
capa = layers
modelo = keras.models.Sequential([
    capa.Input(shape=(32, 32, 3)), # Entrada: imágenes 32x32 a color (3 canales RGB)
    data_augmentation, 
    
    # --- Extracción de características ---
    capa.Conv2D(32, (3, 3), padding='same', activation='relu'), # Patrones básicos (bordes, texturas)
    capa.BatchNormalization(), # Estabiliza y acelera el entrenamiento
    capa.MaxPooling2D(2, 2),   # Reduce la resolución a la mitad
    
    capa.Conv2D(64, (3, 3), padding='same', activation='relu'), # Formas más complejas
    capa.BatchNormalization(),
    capa.MaxPooling2D(2, 2),
    
    capa.Conv2D(64, (3, 3), padding='same', activation='relu'), # Combinación de formas
    capa.BatchNormalization(),
    
    # --- Clasificador ---
    capa.Flatten(),                     # Convierte los mapas de activación en un vector
    capa.Dense(128, activation='relu'), # Capa densa
    capa.BatchNormalization(),
    capa.Dropout(0.4),                  # Desactiva al azar el 40% de las neuronas en cada paso de entrenamiento
    capa.Dense(10)                      # Salida: 10 logits (una por clase de CIFAR-10)
])

# 3. Compilación
modelo.compile(optimizer='adam', # Optimizador
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), # Función de pérdida
                  metrics=['accuracy']) # Métrica

# 4. Callbacks
callbacks = [
    # Detiene el entrenamiento tras 8 épocas sin mejora y restaura los mejores pesos
    EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
    # Guarda en disco el mejor modelo según val_accuracy
    ModelCheckpoint('mejor_modelo.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    # Reduce el learning rate a la mitad si val_loss se estanca durante 3 épocas
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
]

# 5. Ejecución
print("Iniciando entrenamiento...")
historial = modelo.fit(imagenes_entrenamiento, etiquetas_entrenamiento,
                             epochs=40, # Máximo de épocas
                             batch_size=64, # Tamaño de lote
                             validation_data=(imagenes_verificacion, etiquetas_verificacion), # Conjunto de validación
                             callbacks=callbacks)

print("Entrenamiento finalizado. Mejor modelo guardado en mejor_modelo.keras")