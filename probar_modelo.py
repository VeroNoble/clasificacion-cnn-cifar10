import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

print("Cargando el modelo...")
ruta_actual = os.path.dirname(os.path.abspath(__file__))
modelo = load_model(os.path.join(ruta_actual, 'mejor_modelo.keras'))
nombres_clases = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 1. Selección de la imagen mediante un cuadro de diálogo
root = tk.Tk()
root.attributes('-topmost', True)
root.withdraw()
ruta_imagen = filedialog.askopenfilename(title="Seleccionar imagen")

if ruta_imagen:
    # 2. Recorte cuadrado centrado
    img = Image.open(ruta_imagen)
    ancho, alto = img.size
    min_dim = min(ancho, alto)
    izq = (ancho - min_dim) / 2
    sup = (alto - min_dim) / 2
    der = (ancho + min_dim) / 2
    inf = (alto + min_dim) / 2
    img_cuadrada = img.crop((izq, sup, der, inf))

    # 3. Redimensionado a 32x32 y normalización
    img_modelo = img_cuadrada.resize((32, 32))
    img_array = image.img_to_array(img_modelo) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 4. Predicción
    prediccion_cruda = modelo.predict(img_array)
    prediccion_porcentajes = tf.nn.softmax(prediccion_cruda[0]) # Convierte los logits en probabilidades
    
    indice_ganador = np.argmax(prediccion_porcentajes)
    resultado = nombres_clases[indice_ganador]
    porcentaje = prediccion_porcentajes[indice_ganador] * 100

    print(f"\n---> PREDICCIÓN: {resultado.upper()} ({porcentaje:.2f}%) <--- \n")

    # 5. Visualización de la imagen con la clase predicha y su probabilidad
    plt.imshow(img_cuadrada)
    plt.title(f"Predicción: {resultado.upper()} ({porcentaje:.2f}%)")
    plt.axis('off')
    plt.show()
else:
    print("No se seleccionó ninguna imagen.")