import cv2
import numpy as np

def _preparar_para_logica(img_base, img_secundaria):
    """
    Ajusta la img_secundaria para que coincida en dimensiones y canales
    con la img_base.
    """
    if img_base is None or img_secundaria is None:
        raise ValueError("Se requieren dos imágenes para esta operación.")

    # 1. Redimensionar img_secundaria al tamaño de img_base (ancho, alto)
    # shape[1] es ancho, shape[0] es alto
    alto, ancho = img_base.shape[:2]
    img_sec_resized = cv2.resize(img_secundaria, (ancho, alto))

    # 2. Igualar canales (Color vs Escala de Grises)
    base_es_gris = len(img_base.shape) == 2
    sec_es_gris = len(img_sec_resized.shape) == 2

    if base_es_gris and not sec_es_gris:
        # Base gris, Secundaria color -> Convertir secundaria a gris
        img_sec_final = cv2.cvtColor(img_sec_resized, cv2.COLOR_BGR2GRAY)
    elif not base_es_gris and sec_es_gris:
        # Base color, Secundaria gris -> Convertir secundaria a BGR
        img_sec_final = cv2.cvtColor(img_sec_resized, cv2.COLOR_GRAY2BGR)
    else:
        # Ambos son iguales
        img_sec_final = img_sec_resized

    return img_base, img_sec_final

def operacion_and(img_base, img_secundaria):
    """Operación lógica AND: Mantiene píxeles activos en ambas imágenes."""
    img1, img2 = _preparar_para_logica(img_base, img_secundaria)
    return cv2.bitwise_and(img1, img2)

def operacion_or(img_base, img_secundaria):
    """Operación lógica OR: Mantiene píxeles activos si están en alguna de las dos."""
    img1, img2 = _preparar_para_logica(img_base, img_secundaria)
    return cv2.bitwise_or(img1, img2)

def operacion_xor(img_base, img_secundaria):
    """Operación lógica XOR: Mantiene píxeles donde uno es activo y el otro no."""
    img1, img2 = _preparar_para_logica(img_base, img_secundaria)
    return cv2.bitwise_xor(img1, img2)

def operacion_not(img_base):
    """Operación lógica NOT: Invierte los valores de los píxeles (negativo)."""
    return cv2.bitwise_not(img_base)

