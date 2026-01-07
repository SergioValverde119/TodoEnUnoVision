import cv2
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --- UTILIDADES PARA CREAR LOS MAPAS ---
def crear_lut_desde_matplotlib(cmap_name, colors_list):
    """Crea una Lookup Table (LUT) de 256 colores compatible con OpenCV"""
    # 1. Crear mapa Matplotlib
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors_list, N=256)
    
    # 2. Muestrear 256 valores (0 a 1)
    x = np.linspace(0, 1, 256)
    
    # 3. Obtener colores RGBA (0-1) -> Convertir a RGB (0-255)
    rgba = cmap(x)
    rgb = (rgba[:, :3] * 255).astype(np.uint8)
    
    # 4. Matplotlib es RGB, OpenCV usa BGR -> Invertimos el orden de canales
    bgr = rgb[:, ::-1] 
    
    # 5. Formato exacto para cv2.LUT: (256, 1, 3)
    lut = bgr.reshape((256, 1, 3))
    return lut

# --- DEFINICIÓN DE TUS COLORES ---

# PROPIO 1: Colores pastel por tramos (Step function)
colores_propio1 = [
    (0.00, (0.9, 0.8, 1.0)),    # Violeta claro
    (30/255, (0.9, 0.8, 1.0)),
    (30/255, (0.8, 0.9, 1.0)),  # Azul lavanda
    (60/255, (0.8, 0.9, 1.0)),
    (60/255, (0.8, 1.0, 0.8)),  # Verde menta
    (120/255, (0.8, 1.0, 0.8)),
    (120/255, (1.0, 0.8, 0.9)), # Rosa claro
    (200/255, (1.0, 0.8, 0.9)),
    (200/255, (1.0, 1.0, 0.8)), # Amarillo suave
    (1.0,     (1.0, 1.0, 0.8))
]

# PROPIO 2: Gradiente suave personalizado
colores_propio2 = [
    (0.0, (1.0, 0.9, 0.9)),  # Sombras muy claras
    (0.25, (0.9, 1.0, 0.9)), # Medios verdes claros
    (0.5, (0.9, 0.9, 1.0)),  # Medios azules
    (0.75, (1.0, 0.8, 0.5)), # Luces naranjas suaves
    (1.0, (1.0, 0.6, 0.3))   # Brillos intensos
]

# --- GENERACIÓN DE LAS LUTS ---
# Estas variables son las que importaremos desde fuera
LUT_PROPIO_1 = crear_lut_desde_matplotlib("Propio1", colores_propio1)
LUT_PROPIO_2 = crear_lut_desde_matplotlib("Propio2", colores_propio2)

# --- FUNCIÓN PRINCIPAL DE APLICACIÓN ---
def aplicar_mapa_color(imagen, nombre_mapa):
    """
    Recibe la imagen y el nombre del mapa ('JET', 'PROPIO 1', etc.)
    Retorna la imagen coloreada.
    """
    if imagen is None: return None
    
    # Aseguramos que trabajamos sobre una imagen de intensidad (Grises)
    if len(imagen.shape) == 3:
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen

    # Mapas estándar de OpenCV
    if nombre_mapa == "JET":
        return cv2.applyColorMap(gris, cv2.COLORMAP_JET)
    elif nombre_mapa == "HOT":
        return cv2.applyColorMap(gris, cv2.COLORMAP_HOT)
    elif nombre_mapa == "OCEAN":
        return cv2.applyColorMap(gris, cv2.COLORMAP_OCEAN)
    elif nombre_mapa == "BONE":
        return cv2.applyColorMap(gris, cv2.COLORMAP_BONE)
    elif nombre_mapa == "PINK":
        return cv2.applyColorMap(gris, cv2.COLORMAP_PINK)
    
    # Mapas Personalizados (Usamos las LUTs definidas arriba)
    elif nombre_mapa == "PROPIO 1":
        # cv2.LUT necesita 3 canales de entrada si la salida es color
        bgr_temp = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
        return cv2.LUT(bgr_temp, LUT_PROPIO_1)
        
    elif nombre_mapa == "PROPIO 2":
        bgr_temp = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
        return cv2.LUT(bgr_temp, LUT_PROPIO_2)
        
    return cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
