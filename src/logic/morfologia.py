import cv2
import numpy as np

def convertir_a_grises(imagen):
    if len(imagen.shape) == 3:
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return imagen

# --- Operaciones Básicas ---
def erosion(imagen, kernel_size=5):
    img = convertir_a_grises(imagen)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(img, kernel, iterations=1)

def dilatacion(imagen, kernel_size=5):
    img = convertir_a_grises(imagen)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(img, kernel, iterations=1)

# --- Operaciones Compuestas (Manuales) ---
def apertura_manual(imagen, kernel_size=5):
    """Erosión seguida de Dilatación (implementación manual)"""
    img = convertir_a_grises(imagen)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Paso 1: Erosión
    eroded = cv2.erode(img, kernel, iterations=1)
    # Paso 2: Dilatación
    return cv2.dilate(eroded, kernel, iterations=1)

def cierre_manual(imagen, kernel_size=5):
    """Dilatación seguida de Erosión (implementación manual)"""
    img = convertir_a_grises(imagen)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Paso 1: Dilatación
    dilated = cv2.dilate(img, kernel, iterations=1)
    # Paso 2: Erosión
    return cv2.erode(dilated, kernel, iterations=1)

# --- Operaciones EX (OpenCV optimizado) ---
def apertura_ex(imagen, kernel_size=5):
    """Apertura usando morphologyEx"""
    img = convertir_a_grises(imagen)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

def cierre_ex(imagen, kernel_size=5):
    """Cierre usando morphologyEx"""
    img = convertir_a_grises(imagen)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
