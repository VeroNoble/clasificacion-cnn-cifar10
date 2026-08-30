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

# 1. Tu "botón" de carga (abre el explorador de Windows)
root = tk.Tk()
root.attributes('-topmost', True)
root.withdraw()
ruta_imagen = filedialog.askopenfilename(title="Elegí la foto a detectar:")

if ruta_imagen:
    # 2. Abrir la imagen y recortarla en un cuadrado perfecto
    img = Image.open(ruta_imagen)
    ancho, alto = img.size
    min_dim = min(ancho, alto)
    izq = (ancho - min_dim) / 2
    sup = (alto - min_dim) / 2
    der = (ancho + min_dim) / 2
    inf = (alto + min_dim) / 2
    img_cuadrada = img.crop((izq, sup, der, inf))

    # 3. Achicarla a 32x32 
    img_modelo = img_cuadrada.resize((32, 32))
    img_array = image.img_to_array(img_modelo) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # 4. Predecir y calcular el porcentaje
    prediccion_cruda = modelo.predict(img_array)
    prediccion_porcentajes = tf.nn.softmax(prediccion_cruda[0]) # Convierte los números raros a porcentajes del 0 al 1
    
    indice_ganador = np.argmax(prediccion_porcentajes)
    resultado = nombres_clases[indice_ganador]
    porcentaje = prediccion_porcentajes[indice_ganador] * 100

    print(f"\n---> PREDICCIÓN: {resultado.upper()} ({porcentaje:.2f}%) <--- \n")

    # 5. Mostrar la foto recortada en pantalla con el resultado y el %
    plt.imshow(img_cuadrada)
    plt.title(f"El modelo dice que es un: {resultado.upper()} ({porcentaje:.2f}%)")
    plt.axis('off')
    plt.show()
else:
    print("No elegiste ninguna foto.")