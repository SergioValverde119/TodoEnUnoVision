import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# --- NUEVO: CÁLCULO DE PROPIEDADES ---
def calcular_estadisticas(imagen):
    """
    Calcula propiedades estadísticas del histograma de la imagen.
    Retorna un diccionario con los valores.
    """
    # Trabajamos preferentemente en grises para estadísticas globales
    if len(imagen.shape) == 3:
        img_proc = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        img_proc = imagen

    # Aplanar array
    pixels = img_proc.flatten()
    
    # 1. Media (Brillo promedio)
    media = np.mean(pixels)
    
    # 2. Varianza (Contraste)
    varianza = np.var(pixels)
    desviacion = np.sqrt(varianza)
    
    # 3. Valor Máximo y Mínimo
    val_min = np.min(pixels)
    val_max = np.max(pixels)
    
    # 4. Entropía (Información)
    # Calculamos histograma normalizado (probabilidad)
    hist, _ = np.histogram(pixels, bins=256, range=(0, 256), density=True)
    # Eliminamos ceros para el logaritmo
    hist = hist[hist > 0]
    entropia = -np.sum(hist * np.log2(hist))
    
    stats = {
        "Media (Brillo)": f"{media:.2f}",
        "Desviación Std": f"{desviacion:.2f}",
        "Varianza": f"{varianza:.2f}",
        "Mínimo": f"{val_min}",
        "Máximo": f"{val_max}",
        "Entropía": f"{entropia:.4f}"
    }
    return stats

# --- FUNCIONES ORIGINALES (MANTENIDAS) ---

def calcular_histograma(imagen, modelo_actual="RGB"):
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)

    if modelo_actual == "GRAY" or len(imagen.shape) == 2:
        ax.hist(imagen.ravel(), 256, [0, 256], color='gray')
        ax.set_title("Histograma (Grises)")

    elif modelo_actual == "HSV":
        h, s, v = cv2.split(imagen)
        hist_h = cv2.calcHist([h], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
        hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
        ax.plot(hist_h, color='orange', label='Hue (Matiz)')
        ax.plot(hist_s, color='green', label='Sat (Saturación)')
        ax.plot(hist_v, color='purple', label='Val (Brillo)')
        ax.set_title("Histograma HSV")
        ax.set_xlim([0, 256])
        ax.legend()

    elif modelo_actual == "CMYK":
        img_float = imagen.astype(np.float32) / 255.0
        c, m, y_chn = cv2.split(img_float)
        hist_c = cv2.calcHist([c.astype('float32')], [0], None, [256], [0, 1])
        hist_m = cv2.calcHist([m.astype('float32')], [0], None, [256], [0, 1])
        hist_y = cv2.calcHist([y_chn.astype('float32')], [0], None, [256], [0, 1])
        ax.plot(hist_c, color='cyan', label='Cian')
        ax.plot(hist_m, color='magenta', label='Magenta')
        ax.plot(hist_y, color='yellow', label='Amarillo')
        ax.set_title("Niveles de Tinta (CMY)")
        ax.legend()
        
    else: 
        colores = ('b', 'g', 'r')
        labels = ('Azul', 'Verde', 'Rojo')
        for i, color in enumerate(colores):
            hist = cv2.calcHist([imagen], [i], None, [256], [0, 256])
            ax.plot(hist, color=color, label=labels[i])
        ax.set_title("Histograma RGB")
        ax.legend()
        
    ax.grid(True, alpha=0.3)
    return fig

def separar_canales(imagen, modelo="RGB"):
    if len(imagen.shape) == 2:
        return [("Gris / Binario", imagen)]

    c1, c2, c3 = cv2.split(imagen)
    zeros = np.zeros_like(c1)
    canales_visualizables = []

    if modelo == "HSV":
        ones = np.ones_like(c1) * 255
        h_visual = cv2.merge([c1, ones, ones]) 
        h_visual = cv2.cvtColor(h_visual, cv2.COLOR_HSV2BGR) 
        s_visual = cv2.merge([zeros, zeros, c2]) 
        v_visual = c3 
        canales_visualizables = [
            ("Matiz (H)", h_visual), ("Saturación (S)", s_visual), ("Valor (V)", v_visual)
        ]
    elif modelo == "CMYK":
        c_vis = cv2.merge([c1, c1, zeros])
        m_vis = cv2.merge([c2, zeros, c2])
        y_vis = cv2.merge([zeros, c3, c3])
        canales_visualizables = [
            ("Cian", c_vis), ("Magenta", m_vis), ("Amarillo", y_vis)
        ]
    else: 
        azul_vis = cv2.merge([c1, zeros, zeros])
        verde_vis = cv2.merge([zeros, c2, zeros])
        rojo_vis = cv2.merge([zeros, zeros, c3])
        canales_visualizables = [
            ("Azul (B)", azul_vis), ("Verde (G)", verde_vis), ("Rojo (R)", rojo_vis)
        ]

    return canales_visualizables

def etiquetar_componentes(imagen):
    if len(imagen.shape) > 2:
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, labels = cv2.connectedComponents(binaria)
    colores = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colores[0] = [0, 0, 0] 
    resultado_color = colores[labels]
    return resultado_color, num_labels - 1