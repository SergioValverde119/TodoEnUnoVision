import cv2
import numpy as np

def aplicar_modelo(imagen, modelo):
    """
    Controlador principal para cambios de espacio de color.
    modelos: 'RGB', 'GRAY', 'BINARY', 'HSV', 'CMYK'
    """
    if imagen is None: return None
    
    # 1. Recuperar imagen base en BGR (OpenCV estándar)
    # Si la imagen ya viene en gris (2 dim), la convertimos a BGR primero para estandarizar
    if len(imagen.shape) == 2:
        img_bgr = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = imagen.copy()

    # 2. Aplicar transformación
    if modelo == "RGB":
        # OpenCV usa BGR por defecto, así que "RGB" para visualización es 
        # simplemente volver al estado original de carga.
        return img_bgr

    elif modelo == "GRAY":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    elif modelo == "BINARY":
        # Primero a gris
        gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        # Threshold (umbral) automático usando Otsu
        _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binaria

    elif modelo == "HSV":
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    elif modelo == "CMYK":
        # OpenCV no tiene conversión directa BGR -> CMYK. Hay que hacerlo manual.
        # Normalizar a rango 0-1
        img_float = img_bgr.astype(np.float32) / 255.0
        
        # BGR -> CMY
        # K = 1 - max(R, G, B)
        # C = (1 - R - K) / (1 - K) ... etc.
        # Truco: CMY es simplemente 1 - RGB (invertir canales)
        # Pero ojo: BGR -> RGB primero
        b, g, r = cv2.split(img_float)
        
        # K (Key/Black) es el mínimo valor de luz (1 - max(r,g,b))
        # Pero es más fácil visualmente mostrar la inversión CMY simple
        # Retornaremos la imagen en formato CMY simulado para visualización
        c = 1.0 - r
        m = 1.0 - g
        y = 1.0 - b
        
        # Juntamos para mostrar (aunque los monitores son RGB, esto simula el efecto)
        cmy = cv2.merge((c, m, y))
        return (cmy * 255).astype(np.uint8)
        
    return imagen
