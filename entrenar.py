import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Mantiene la terminal limpia de advertencias

# =====================================================================
# 1. CÓDIGO BASE DEL PROFESOR
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


#def mostrar():
#  plt.figure(figsize=(5,5))
#  for i in range(10):
#    plt.subplot(4,4,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(imagenes_entrenamiento[i], cmap=plt.cm.binary)
#    plt.xlabel(nombres_clases[etiquetas_entrenamiento[i][0]])
#  plt.show()
#mostrar()

# =====================================================================
# ARQUITECTURA Y ENTRENAMIENTO DEL MODELO
# =====================================================================

print("Construyendo arquitectura optimizada...")

# 1. Aumento de datos: Evita que el modelo memorice (sobreajuste)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"), # Espeja la imagen aleatoriamente
    layers.RandomRotation(0.1),      # Rota la imagen hasta un 10%
])

# 2. Construcción de la Red Neuronal
capa = layers
modelo = keras.models.Sequential([
    capa.Input(shape=(32, 32, 3)), # Entrada: imágenes 32x32 a color (3 canales RGB)
    data_augmentation, 
    
    # --- Extracción de características (Filtros visuales) ---
    capa.Conv2D(32, (3, 3), padding='same', activation='relu'), # Detecta patrones básicos
    capa.BatchNormalization(), # Estabiliza y acelera la matemática interna
    capa.MaxPooling2D(2, 2),   # Achica la imagen a la mitad reteniendo lo importante
    
    capa.Conv2D(64, (3, 3), padding='same', activation='relu'), # Detecta formas complejas
    capa.BatchNormalization(),
    capa.MaxPooling2D(2, 2),
    
    capa.Conv2D(64, (3, 3), padding='same', activation='relu'), # Combina las formas
    capa.BatchNormalization(),
    
    # --- Clasificador final (El cerebro de la red) ---
    capa.Flatten(),                     # Aplana los datos a un vector de una sola dimensión
    capa.Dense(128, activation='relu'), # Neuronas conectadas para tomar la decisión
    capa.BatchNormalization(),
    capa.Dropout(0.4),                  # "Apaga" el 40% de las neuronas para forzar un aprendizaje real
    capa.Dense(10)                      # Salida: 10 opciones (las 10 categorías de CIFAR-10)
])

# 3. Compilación: Reglas de cómo va a aprender
modelo.compile(optimizer='adam', # Algoritmo que ajusta los pesos
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), # Mide el error
                  metrics=['accuracy']) # Mide el porcentaje de acierto

# 4. Callbacks: Automatizaciones de seguridad
callbacks = [
    # Frena el entrenamiento si pasan 8 épocas sin mejorar y restaura la mejor versión
    EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
    # Guarda físicamente el archivo del mejor modelo en tu disco
    ModelCheckpoint('mejor_modelo.keras', monitor='val_accuracy', save_best_only=True, verbose=1),
    # Si el error se estanca, achica el paso de aprendizaje para hacer ajustes más finos
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
]

# 5. Ejecución
print("Iniciando entrenamiento...")
historial = modelo.fit(imagenes_entrenamiento, etiquetas_entrenamiento,
                             epochs=40, # Máximo de vueltas permitidas
                             batch_size=64, # Procesa las imágenes en grupos de 64
                             validation_data=(imagenes_verificacion, etiquetas_verificacion), # Datos de prueba
                             callbacks=callbacks)

print("Entrenamiento finalizado. El mejor modelo está guardado.")