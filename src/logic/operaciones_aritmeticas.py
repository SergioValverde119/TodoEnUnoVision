import cv2
import numpy as np

def _preparar_para_operacion(img_base, img_secundaria):
    """
    Iguala dimensiones y canales de la imagen secundaria a la base.
    """
    if img_base is None or img_secundaria is None:
        raise ValueError("Se requieren dos imágenes.")

    # 1. Redimensionar
    alto, ancho = img_base.shape[:2]
    img_sec_resized = cv2.resize(img_secundaria, (ancho, alto))

    # 2. Igualar canales
    base_es_gris = len(img_base.shape) == 2
    sec_es_gris = len(img_sec_resized.shape) == 2

    if base_es_gris and not sec_es_gris:
        img_sec_final = cv2.cvtColor(img_sec_resized, cv2.COLOR_BGR2GRAY)
    elif not base_es_gris and sec_es_gris:
        img_sec_final = cv2.cvtColor(img_sec_resized, cv2.COLOR_GRAY2BGR)
    else:
        img_sec_final = img_sec_resized

    return img_base, img_sec_final

# ==========================================
# OPERACIONES CON IMÁGENES (Pixel a Pixel)
# ==========================================

def suma_imagenes(img1, img2):
    i1, i2 = _preparar_para_operacion(img1, img2)
    return cv2.add(i1, i2)

def resta_imagenes(img1, img2):
    i1, i2 = _preparar_para_operacion(img1, img2)
    return cv2.subtract(i1, i2)

def multiplicacion_imagenes(img1, img2):
    i1, i2 = _preparar_para_operacion(img1, img2)
    # Se usa una escala de 1/255.0 a veces para blending, 
    # pero aquí haremos multiplicación directa con saturación.
    return cv2.multiply(i1, i2)

def division_imagenes(img1, img2):
    i1, i2 = _preparar_para_operacion(img1, img2)
    return cv2.divide(i1, i2)

# ==========================================
# OPERACIONES CON ESCALARES 
# ==========================================

def suma_escalar(img, valor):
    """Aumenta el brillo"""
    # Se crea un array/tupla para sumar a todos los canales
    val_tuple = (valor, valor, valor, 0) if len(img.shape) == 3 else valor
    return cv2.add(img, val_tuple)

def resta_escalar(img, valor):
    """Disminuye el brillo"""
    val_tuple = (valor, valor, valor, 0) if len(img.shape) == 3 else valor
    return cv2.subtract(img, val_tuple)

def multiplicacion_escalar(img, valor):
    """Aumenta el contraste (si valor > 1)"""
    # cv2.multiply con escalar requiere que el escalar sea del mismo tipo o usar scale
    # Opción robusta: convertir a float, multiplicar y volver a uint8
    res = cv2.multiply(img.astype(float), float(valor))
    # Saturar manualmente y convertir
    res = np.clip(res, 0, 255).astype(np.uint8)
    return res

def division_escalar(img, valor):
    """Disminuye el contraste (si valor > 1)"""
    if valor == 0: return img
    # Inverso de multiplicación
    res = cv2.divide(img.astype(float), float(valor))
    res = np.clip(res, 0, 255).astype(np.uint8)
    return res

def inversion_aritmetica(img):
    """
    Invierte los valores de los píxeles (Negativo).
    Fórmula: 255 - pixel
    """
    img_blanca = np.full(img.shape, 255, dtype=np.uint8)
    return cv2.subtract(img_blanca, img)