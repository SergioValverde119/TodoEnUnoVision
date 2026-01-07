import cv2
import numpy as np
import scipy.ndimage as ndimage

def convertir_a_grises(imagen):
    """Convierte a escala de grises si es necesario"""
    if len(imagen.shape) == 3:
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return imagen

# ==========================================
# FILTROS DE SUAVIZADO (RUIDO)
# ==========================================

def filtro_promedio(imagen, kernel_size=3):
    """Filtro de media o promedio (Blur)"""
    return cv2.blur(imagen, (kernel_size, kernel_size))

def filtro_mediana(imagen, kernel_size=3):
    """Filtro de mediana (ideal para sal y pimienta)"""
    # kernel_size debe ser impar
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.medianBlur(imagen, k)

def filtro_gaussiano(imagen, kernel_size=3):
    """Filtro Gaussiano"""
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(imagen, (k, k), 0)

def filtro_maximo(imagen, kernel_size=3):
    """Filtro de máximo (elimina puntos negros/pimienta)"""
    # Funciona mejor en escala de grises para visualizar
    img = convertir_a_grises(imagen)
    return ndimage.maximum_filter(img, size=kernel_size)

def filtro_minimo(imagen, kernel_size=3):
    """Filtro de mínimo (elimina puntos blancos/sal)"""
    img = convertir_a_grises(imagen)
    return ndimage.minimum_filter(img, size=kernel_size)


# ==========================================
# FILTROS DE DETECCIÓN DE BORDES
# ==========================================

def filtro_sobel(imagen):
    """Operador Sobel (Magnitud de gradientes X e Y)"""
    img = convertir_a_grises(imagen)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calcular magnitud
    magnitud = cv2.magnitude(sobelx, sobely)
    
    # Normalizar a 0-255 y convertir a uint8
    magnitud = cv2.normalize(magnitud, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(magnitud)

def filtro_prewitt(imagen):
    """Operador Prewitt"""
    img = convertir_a_grises(imagen)
    
    # Kernels de Prewitt
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
    
    # Aplicar filtros
    prewittx = cv2.filter2D(img, -1, kernelx)
    prewitty = cv2.filter2D(img, -1, kernely)
    
    # Combinar (suma ponderada aproximada o magnitud)
    # Usamos addWeighted para simular la combinación visual
    bordes = cv2.addWeighted(prewittx, 0.5, prewitty, 0.5, 0)
    return bordes

def filtro_roberts(imagen):
    """Operador Roberts"""
    img = convertir_a_grises(imagen)
    
    # Kernels de Roberts
    kernelx = np.array([[1, 0], [0, -1]], dtype=np.float32)
    kernely = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    
    robertsx = cv2.filter2D(img, -1, kernelx)
    robertsy = cv2.filter2D(img, -1, kernely)
    
    bordes = cv2.addWeighted(robertsx, 0.5, robertsy, 0.5, 0)
    return bordes

def filtro_canny(imagen):
    """Detector de bordes Canny"""
    img = convertir_a_grises(imagen)
    # Umbrales estándar, se pueden ajustar
    return cv2.Canny(img, 100, 200)

def filtro_laplaciano(imagen):
    """Filtro Laplaciano"""
    img = convertir_a_grises(imagen)
    # Usamos CV_64F para capturar bordes negativos y luego valor absoluto
    lap = cv2.Laplacian(img, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    return lap

def filtro_kirsch(imagen):
    """Operador Kirsch (Máximo de 8 direcciones)"""
    img = convertir_a_grises(imagen)
    
    # Las 8 máscaras de Kirsch
    kernels = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),      # N
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),      # NE
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),      # E
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),      # SE
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),      # S
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),      # SW
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),      # W
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]])       # NW
    ]
    
    # Aplicar todos los kernels y tomar el máximo en cada píxel
    respuestas = [cv2.filter2D(img, -1, k) for k in kernels]
    
    # Combinar tomando el máximo valor pixel a pixel
    bordes = np.max(respuestas, axis=0)
    return bordes
