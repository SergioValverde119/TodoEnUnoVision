import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

def calcular_histograma(imagen, modelo_actual="RGB"):
    fig = Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot(111)

    if modelo_actual == "GRAY" or len(imagen.shape) == 2:
        ax.hist(imagen.ravel(), 256, [0, 256], color='gray')
        ax.set_title("Histograma (Grises)")

    elif modelo_actual == "HSV":
        # Separamos canales H, S, V
        h, s, v = cv2.split(imagen)
        
        # Graficamos cada uno con su color representativo
        # H (Matiz): 0-179 en OpenCV, S/V: 0-255
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
        # Asumimos que la imagen entrante es una visualización BGR simulada de CMY
        # Lo ideal sería tener la matriz CMYK original, pero si no, la recalculamos
        # para mostrar el gráfico correcto.
        img_float = imagen.astype(np.float32) / 255.0
        # Invertimos BGR (que simula CMY) para obtener "tintas" aproximadas
        # B de la imagen = Y del modelo
        # G de la imagen = M del modelo
        # R de la imagen = C del modelo
        c, m, y_chn = cv2.split(img_float)
        
        hist_c = cv2.calcHist([c.astype('float32')], [0], None, [256], [0, 1])
        hist_m = cv2.calcHist([m.astype('float32')], [0], None, [256], [0, 1])
        hist_y = cv2.calcHist([y_chn.astype('float32')], [0], None, [256], [0, 1])
        
        ax.plot(hist_c, color='cyan', label='Cian')
        ax.plot(hist_m, color='magenta', label='Magenta')
        ax.plot(hist_y, color='yellow', label='Amarillo')
        ax.set_title("Niveles de Tinta (CMY)")
        ax.legend()
        
    else: # RGB por defecto
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
    """
    Separa los canales y los prepara para visualización CORRECTA según el modelo.
    """
    if len(imagen.shape) == 2:
        return [("Gris / Binario", imagen)]

    # Separamos los 3 canales crudos
    c1, c2, c3 = cv2.split(imagen)
    zeros = np.zeros_like(c1)

    canales_visualizables = []

    if modelo == "HSV":
        # Canal H (Matiz): Lo mostramos con un mapa de color HSV para que se entienda
        # Truco: Creamos una imagen HSV donde S=255 y V=255 para ver el color puro del matiz
        ones = np.ones_like(c1) * 255
        h_visual = cv2.merge([c1, ones, ones]) 
        h_visual = cv2.cvtColor(h_visual, cv2.COLOR_HSV2BGR) # Convertimos a BGR para que Qt lo entienda
        
        # Canal S (Saturación): Escala de grises o Rojizo (blanco=muy saturado)
        s_visual = cv2.merge([zeros, zeros, c2]) # Se verá rojo intenso donde haya saturación
        
        # Canal V (Valor): Escala de grises (es brillo)
        v_visual = c3 

        canales_visualizables = [
            ("Matiz (H) - Color Puro", h_visual),
            ("Saturación (S) - Intensidad", s_visual), # O puedes usar c2 directo si prefieres gris
            ("Valor (V) - Brillo", v_visual)
        ]

    elif modelo == "CMYK":
        # En nuestra implementación, CMYK es una simulación visual donde B=Y, G=M, R=C
        # Porque lo convertimos invirtiendo canales.
        # c1 es Cian, c2 es Magenta, c3 es Amarillo
        
        # Para ver Cian: Dejamos solo el canal azul+verde o invertimos el rojo?
        # Visualmente en pantalla: Cian = Verde + Azul.
        cian_vis = cv2.merge([c1, c1, zeros]) # Mezcla rara para simular.
        # Mejor enfoque: Mostrar la "tinta" negra sobre papel blanco
        # Invertimos el canal para que lo "oscuro" sea la tinta
        
        # Simulación simple de tintas:
        # CIAN puro en pantalla es (255, 255, 0) en BGR
        # MAGENTA puro es (255, 0, 255) en BGR
        # AMARILLO puro es (0, 255, 255) en BGR
        
        # Pero tus datos ya vienen transformados. Asumimos que c1=C, c2=M, c3=Y
        # Para visualizar C, mostramos c1 pintado de cian.
        # ¿Cómo pintar una imagen de grises de color cian? 
        # C = B+G. Entonces ponemos el valor de c1 en los canales B y G.
        c_vis = cv2.merge([c1, c1, zeros])
        m_vis = cv2.merge([c2, zeros, c2])
        y_vis = cv2.merge([zeros, c3, c3])

        canales_visualizables = [
            ("Canal Cian", c_vis),
            ("Canal Magenta", m_vis),
            ("Canal Amarillo", y_vis)
        ]

    else: # RGB (o default)
        # BGR Estándar de OpenCV
        # c1=Azul, c2=Verde, c3=Rojo
        azul_vis = cv2.merge([c1, zeros, zeros])
        verde_vis = cv2.merge([zeros, c2, zeros])
        rojo_vis = cv2.merge([zeros, zeros, c3])
        
        canales_visualizables = [
            ("Canal Azul (B)", azul_vis),
            ("Canal Verde (G)", verde_vis),
            ("Canal Rojo (R)", rojo_vis)
        ]

    return canales_visualizables


# --- Agregar al final de src/logic/analisis.py ---
def etiquetar_componentes(imagen):
    """
    Recibe una imagen (preferiblemente binaria).
    Retorna la imagen coloreada con etiquetas aleatorias.
    """
    # 1. Asegurar que sea gris/binaria
    if len(imagen.shape) > 2:
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen

    # 2. Binarizar si no lo está (Otsu ayuda a limpiar ruido)
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3. Calcular componentes conexas
    # num_labels: cantidad de objetos encontrados (incluyendo fondo)
    # labels: matriz donde cada pixel tiene el ID de su objeto (0, 1, 2...)
    num_labels, labels = cv2.connectedComponents(binaria)

    # 4. Colorear (Mapear etiquetas a colores)
    # Creamos una paleta de colores aleatorios
    # (num_labels) colores de 3 canales (RGB)
    colores = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    
    # El fondo (etiqueta 0) lo forzamos a negro para que se vea limpio
    colores[0] = [0, 0, 0] 

    # Aplicamos los colores a la matriz de etiquetas
    resultado_color = colores[labels]
    
    return resultado_color, num_labels - 1 # Restamos 1 para no contar el fondo
