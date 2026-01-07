import cv2
import os
import sys

# --- SOLUCIÓN ERROR IMPORT ---
# Añadir la raíz del proyecto al path para que Python encuentre 'src'
# Se asume que este archivo está en: .../Image-Analysis-App/src/ui/ventana.py
current_dir = os.path.dirname(os.path.abspath(__file__)) # directorio actual (src/ui)
project_root = os.path.abspath(os.path.join(current_dir, "..", "..")) # subir 2 niveles (Image-Analysis-App)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# -----------------------------

from datetime import datetime
from PyQt6.QtWidgets import (
    QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, 
    QWidget, QFileDialog, QMessageBox, QSizePolicy, QInputDialog,
    QFrame, QToolBox, QMenu  # <--- AÑADIDO QMenu AQUÍ
)
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtCore import Qt

# Importamos módulos de lógica
from src.logic.gestor_estado import GestorEstado
from src.ui.ventanas_aux import VentanaHistograma, VentanaCanales
from src.logic import analisis
from src.logic import operaciones_aritmeticas
from src.logic import operaciones_logicas
from src.logic import colores
from src.logic import mapas
from src.logic import filtros
from src.logic import morfologia
from src.logic import frecuencia

class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Visión Artificial Studio - [Tu Apellido]") # ¡Personaliza esto!
        self.resize(1200, 800)
        
        # Estilo Global (Dark Theme Moderno)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                color: #e0e0e0;
                font-family: 'Segoe UI', sans-serif;
                font-size: 14px;
            }
            /* Estilo para los botones del menú */
            QPushButton {
                background-color: #2d2d2d;
                border: none;
                border-radius: 5px;
                padding: 8px 15px;
                text-align: left;
                margin: 2px;
            }
            QPushButton:hover {
                background-color: #3e3e3e;
                border-left: 3px solid #0078d7;
            }
            QPushButton:pressed {
                background-color: #0078d7;
                color: white;
            }
            /* Títulos de secciones */
            QLabel#TituloSeccion {
                color: #0078d7;
                font-weight: bold;
                font-size: 16px;
                margin-top: 10px;
                margin-bottom: 5px;
                text-transform: uppercase;
            }
            /* Visores */
            QLabel#Visor {
                background-color: #252526;
                border: 2px dashed #444;
                border-radius: 8px;
            }
            QLabel#VisorResultado {
                background-color: #252526;
                border: 2px solid #0078d7;
                border-radius: 8px;
            }
            /* Toolbox (Acordeón) */
            QToolBox::tab {
                background: #252526;
                border-radius: 5px;
                color: #bbb;
                font-weight: bold;
            }
            QToolBox::tab:selected {
                background: #0078d7;
                color: white;
            }
        """)

        # Configuración de Rutas
        base_dir = os.getcwd()
        self.ruta_data = os.path.join(base_dir, "data")
        self.ruta_salidas = os.path.join(base_dir, "salidas")
        os.makedirs(self.ruta_salidas, exist_ok=True)
        
        # Estado
        self.gestor = GestorEstado()
        self.imagen_original = None
        self.imagen_mostrada = None

        # Inicializar UI
        self.init_ui_moderna()

    def init_ui_moderna(self):
        # Widget Central
        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        
        # Layout Principal: [ Menu Lateral | Visores ]
        layout_principal = QHBoxLayout(widget_central)
        layout_principal.setContentsMargins(0, 0, 0, 0)
        layout_principal.setSpacing(0)

        # --- 1. PANEL LATERAL (Barra Vertical) ---
        panel_lateral = QFrame()
        panel_lateral.setFixedWidth(320) # Un poco más ancho para nombres largos
        panel_lateral.setStyleSheet("background-color: #181818; border-right: 1px solid #333;")
        layout_lateral = QVBoxLayout(panel_lateral)
        
        # Título de la App
        lbl_app = QLabel("VISION STUDIO")
        lbl_app.setStyleSheet("font-size: 22px; font-weight: bold; color: white; margin: 20px 10px;")
        layout_lateral.addWidget(lbl_app)

        # Botones Principales (Archivo)
        layout_arch = QHBoxLayout()
        btn_abrir = QPushButton("Abrir")
        btn_abrir.setStyleSheet("background-color: #0078d7; color: white; font-weight: bold; text-align: center;")
        btn_abrir.clicked.connect(self.cargar_imagen)
        
        btn_guardar = QPushButton("Guardar")
        btn_guardar.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; text-align: center;")
        btn_guardar.clicked.connect(self.guardar_imagen)
        
        layout_arch.addWidget(btn_abrir)
        layout_arch.addWidget(btn_guardar)
        layout_lateral.addLayout(layout_arch)
        
        # Historial (Undo/Redo)
        layout_hist = QHBoxLayout()
        btn_undo = QPushButton("↺ Deshacer")
        btn_undo.clicked.connect(self.accion_atras)
        btn_redo = QPushButton("↻ Rehacer")
        btn_redo.clicked.connect(self.accion_adelante)
        layout_hist.addWidget(btn_undo)
        layout_hist.addWidget(btn_redo)
        layout_lateral.addLayout(layout_hist)

        # --- TOOLBOX (Menú Acordeón) ---
        self.toolbox = QToolBox()
        self.toolbox.setStyleSheet("QToolBox { background-color: #181818; }")
        
        # > Sección 1: Operaciones Puntuales (Pre-procesamiento)
        # Sigue el orden de "Imagen 1" de tu lista
        page_puntuales = QWidget()
        lay_punt = QVBoxLayout(page_puntuales)
        
        lbl_ajustes = QLabel("Ajustes Básicos"); lbl_ajustes.setObjectName("TituloSeccion")
        lay_punt.addWidget(lbl_ajustes)
        self.crear_boton("Ajuste Brillo/Contraste", lambda: self.gestionar_aritmetica("SUMA"), lay_punt)
        self.crear_boton("Invertir (Negativo)", lambda: self.gestionar_aritmetica("INV"), lay_punt)
        
        lbl_conv = QLabel("Conversiones"); lbl_conv.setObjectName("TituloSeccion")
        lay_punt.addWidget(lbl_conv)
        self.crear_boton("Escala de Grises", lambda: self.aplicar_modelo("GRAY"), lay_punt)
        self.crear_boton("Binarización (Otsu)", lambda: self.aplicar_modelo("BINARY"), lay_punt)
        self.crear_boton("Modelos Color (HSV/CMYK)", self.menu_color_popup, lay_punt) # Movido aquí
        
        lbl_log = QLabel("Lógicas"); lbl_log.setObjectName("TituloSeccion")
        lay_punt.addWidget(lbl_log)
        self.crear_boton("Operación NOT", lambda: self.aplicar_logica("NOT"), lay_punt)
        self.crear_boton("AND / OR / XOR", self.menu_logicas_popup, lay_punt) # Movido aquí
        
        lay_punt.addStretch()
        self.toolbox.addItem(page_puntuales, "1. Operaciones Puntuales")

        # > Sección 2: Histograma y Análisis
        page_ana = QWidget()
        lay_ana = QVBoxLayout(page_ana)
        
        lbl_histo = QLabel("Histograma"); lbl_histo.setObjectName("TituloSeccion")
        lay_ana.addWidget(lbl_histo)
        self.crear_boton("Ver Histograma", self.mostrar_histograma_simple, lay_ana)
        self.crear_boton("Propiedades Estadísticas", self.mostrar_histograma_completo, lay_ana)
        self.crear_boton("Ecualizar Histograma", self.ecualizar_histograma, lay_ana)
        
        lbl_comp = QLabel("Componentes"); lbl_comp.setObjectName("TituloSeccion")
        lay_ana.addWidget(lbl_comp)
        self.crear_boton("Separar Canales", self.mostrar_canales, lay_ana)
        self.crear_boton("Etiquetado C.C. (Conteo)", self.mostrar_componentes, lay_ana)
        
        lay_ana.addStretch()
        self.toolbox.addItem(page_ana, "2. Análisis e Histogramas")

        # > Sección 3: Operaciones de Vecindad (Filtros Espaciales)
        # Sigue el orden de "Imagen 2"
        page_vec = QWidget()
        lay_vec = QVBoxLayout(page_vec)
        
        lbl_lin = QLabel("Filtros Lineales"); lbl_lin.setObjectName("TituloSeccion")
        lay_vec.addWidget(lbl_lin)
        self.crear_boton("Pasa Bajas (Promedio/Gauss)", self.menu_paso_bajas_popup, lay_vec)
        self.crear_boton("Pasa Altas (Bordes)", self.menu_paso_altas_popup, lay_vec)
        
        lbl_nolin = QLabel("Filtros No Lineales (Orden)"); lbl_nolin.setObjectName("TituloSeccion")
        lay_vec.addWidget(lbl_nolin)
        self.crear_boton("Mediana (Sal/Pimienta)", lambda: self.aplicar_filtro("Mediana"), lay_vec)
        self.crear_boton("Mínimo / Máximo", self.menu_min_max_popup, lay_vec)
        
        lay_vec.addStretch()
        self.toolbox.addItem(page_vec, "3. Operaciones de Vecindad")

        # > Sección 4: Morfología Matemática
        page_morfo = QWidget()
        lay_morfo = QVBoxLayout(page_morfo)
        
        lbl_basica = QLabel("Operaciones Básicas"); lbl_basica.setObjectName("TituloSeccion")
        lay_morfo.addWidget(lbl_basica)
        self.crear_boton("Erosión", lambda: self.aplicar_morfologia("Erosión"), lay_morfo)
        self.crear_boton("Dilatación", lambda: self.aplicar_morfologia("Dilatación"), lay_morfo)
        
        lbl_compuestas = QLabel("Compuestas"); lbl_compuestas.setObjectName("TituloSeccion")
        lay_morfo.addWidget(lbl_compuestas)
        self.crear_boton("Apertura (Limpiar Ruido)", lambda: self.aplicar_morfologia("Apertura"), lay_morfo)
        self.crear_boton("Cierre (Cerrar Huecos)", lambda: self.aplicar_morfologia("Cierre"), lay_morfo)
        
        lay_morfo.addStretch()
        self.toolbox.addItem(page_morfo, "4. Morfología Matemática")

        # > Sección 5: Filtros Frecuenciales
        page_freq = QWidget()
        lay_freq = QVBoxLayout(page_freq)
        self.crear_boton("Pasa Bajas Ideal (FFT)", lambda: self.aplicar_filtro_frec("PASA_BAJAS"), lay_freq)
        self.crear_boton("Pasa Altas Ideal (FFT)", lambda: self.aplicar_filtro_frec("PASA_ALTAS"), lay_freq)
        lay_freq.addStretch()
        self.toolbox.addItem(page_freq, "5. Filtros Frecuenciales")
        
        layout_lateral.addWidget(self.toolbox)
        
        # Información de estado al pie del panel
        self.lbl_info = QLabel("Listo")
        self.lbl_info.setStyleSheet("color: #666; font-size: 11px; margin-top: 10px;")
        layout_lateral.addWidget(self.lbl_info)

        layout_principal.addWidget(panel_lateral)

        # --- 2. ZONA DE VISORES (Derecha) ---
        area_visores = QWidget()
        area_visores.setStyleSheet("background-color: #1e1e1e;")
        layout_visores = QVBoxLayout(area_visores)
        
        # Se elimina el título "MESA DE TRABAJO" que no hacía nada

        # Contenedor de imágenes
        layout_imgs = QHBoxLayout()
        
        # Visor Izquierda (Original)
        self.visor_izq = QLabel("Carga una imagen...")
        self.visor_izq.setObjectName("Visor")
        self.visor_izq.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.visor_izq.setMinimumSize(400, 400)
        self.visor_izq.setScaledContents(False) # Mantener aspecto

        # Visor Derecha (Resultado)
        self.visor_der = QLabel("Resultado")
        self.visor_der.setObjectName("VisorResultado")
        self.visor_der.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.visor_der.setMinimumSize(400, 400)
        self.visor_der.setScaledContents(False)

        policy = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.visor_izq.setSizePolicy(policy)
        self.visor_der.setSizePolicy(policy)

        layout_imgs.addWidget(self.visor_izq)
        layout_imgs.addWidget(self.visor_der)
        
        layout_visores.addLayout(layout_imgs)
        
        # Inicialmente ocultamos la izquierda para modo "Simple"
        self.visor_izq.hide()

        layout_principal.addWidget(area_visores)

    # --- Ayudantes de UI ---
    def crear_boton(self, texto, funcion, layout):
        btn = QPushButton(texto)
        btn.clicked.connect(funcion)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(btn)
        return btn

    # --- Popups para agrupar filtros y ahorrar espacio ---
    def menu_logicas_popup(self):
        menu = QMenu(self)
        self.estilizar_menu(menu)
        menu.addAction("Operación AND", lambda: self.aplicar_logica("AND"))
        menu.addAction("Operación OR", lambda: self.aplicar_logica("OR"))
        menu.addAction("Operación XOR", lambda: self.aplicar_logica("XOR"))
        menu.exec(QIcon(), self.cursor().pos())

    def menu_color_popup(self):
        menu = QMenu(self)
        self.estilizar_menu(menu)
        menu.addAction("Modelo RGB", lambda: self.aplicar_modelo("RGB"))
        menu.addAction("Modelo HSV", lambda: self.aplicar_modelo("HSV"))
        menu.addAction("Modelo CMYK", lambda: self.aplicar_modelo("CMYK"))
        menu.exec(QIcon(), self.cursor().pos())

    def menu_paso_bajas_popup(self):
        menu = QMenu(self)
        self.estilizar_menu(menu)
        menu.addAction("Promedio (Media)", lambda: self.aplicar_filtro("Promedio"))
        menu.addAction("Gaussiano", lambda: self.aplicar_filtro("Gaussiano"))
        menu.exec(QIcon(), self.cursor().pos())

    def menu_paso_altas_popup(self):
        menu = QMenu(self)
        self.estilizar_menu(menu)
        menu.addAction("Canny (Óptimo)", lambda: self.aplicar_filtro("Canny"))
        menu.addAction("Sobel", lambda: self.aplicar_filtro("Sobel"))
        menu.addAction("Prewitt", lambda: self.aplicar_filtro("Prewitt"))
        menu.addAction("Laplaciano", lambda: self.aplicar_filtro("Laplaciano"))
        menu.exec(QIcon(), self.cursor().pos())

    def menu_min_max_popup(self):
        menu = QMenu(self)
        self.estilizar_menu(menu)
        menu.addAction("Mínimo", lambda: self.aplicar_filtro("Mínimo"))
        menu.addAction("Máximo", lambda: self.aplicar_filtro("Máximo"))
        menu.exec(QIcon(), self.cursor().pos())

    def estilizar_menu(self, menu):
        menu.setStyleSheet("""
            QMenu { background-color: #2d2d2d; color: white; border: 1px solid #444; }
            QMenu::item { padding: 5px 20px; }
            QMenu::item:selected { background-color: #0078d7; }
        """)

    # ==========================================
    #              LÓGICA (MÉTODOS)
    # ==========================================

    def ecualizar_histograma(self):
        if self.imagen_mostrada is None: return
        self.gestor.guardar_estado(self.imagen_mostrada)
        
        if len(self.imagen_mostrada.shape) == 3:
            img_gray = cv2.cvtColor(self.imagen_mostrada, cv2.COLOR_BGR2GRAY)
            res = cv2.equalizeHist(img_gray)
            self.imagen_mostrada = res
            self.modelo_actual = "GRAY"
        else:
            self.imagen_mostrada = cv2.equalizeHist(self.imagen_mostrada)
        
        self.actualizar_visores()
        self.lbl_info.setText("Info: Histograma ecualizado.")

    def mostrar_histograma_simple(self):
        self._mostrar_hist(con_stats=False)

    def mostrar_histograma_completo(self):
        self._mostrar_hist(con_stats=True)

    def _mostrar_hist(self, con_stats=False):
        if self.imagen_mostrada is None:
            QMessageBox.warning(self, "Aviso", "Primero carga una imagen.")
            return
        if not hasattr(self, 'modelo_actual'): self.modelo_actual = "RGB"
        fig = analisis.calcular_histograma(self.imagen_mostrada, self.modelo_actual)
        stats = None
        if con_stats:
            stats = analisis.calcular_estadisticas(self.imagen_mostrada)
        dialogo = VentanaHistograma(fig, stats)
        dialogo.exec() 

    def aplicar_filtro_frec(self, tipo):
        if self.imagen_mostrada is None: return
        self.gestor.guardar_estado(self.imagen_mostrada)
        try:
            res = frecuencia.aplicar_filtro_ideal(self.imagen_mostrada, tipo, radio_corte=40)
            self.imagen_mostrada = res
            self.modelo_actual = "GRAY"
            self.actualizar_visores()
            self.lbl_info.setText(f"Info: Filtro Frecuencial {tipo} aplicado.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def cargar_imagen(self):
        archivo, _ = QFileDialog.getOpenFileName(self, "Abrir imagen", self.ruta_data, "Imagenes (*.png *.jpg *.bmp *.tif)")
        if archivo:
            img = cv2.imread(archivo)
            if img is None:
                QMessageBox.critical(self, "Error", "No se pudo leer la imagen.")
                return
            self.gestor.reiniciar()
            self.modelo_actual = "RGB"
            self.imagen_original = img
            self.imagen_mostrada = img.copy()
            self.actualizar_visores()
            self.lbl_info.setText(f"Cargado: {os.path.basename(archivo)}")

    def guardar_imagen(self):
        if self.imagen_mostrada is None: return
        nombre_archivo = f"resultado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        ruta_completa = os.path.join(self.ruta_salidas, nombre_archivo)
        cv2.imwrite(ruta_completa, self.imagen_mostrada)
        QMessageBox.information(self, "Guardado", f"Imagen guardada en:\n{ruta_completa}")

    def accion_atras(self):
        imagen_anterior = self.gestor.deshacer(self.imagen_mostrada)
        if imagen_anterior is not None:
            self.imagen_mostrada = imagen_anterior
            self.modelo_actual = "RGB"
            self.actualizar_visores()
            self.lbl_info.setText("Acción deshecha")

    def accion_adelante(self):
        imagen_siguiente = self.gestor.rehacer(self.imagen_mostrada)
        if imagen_siguiente is not None:
            self.imagen_mostrada = imagen_siguiente
            self.modelo_actual = "RGB"
            self.actualizar_visores()
            self.lbl_info.setText("Acción rehecha")

    def actualizar_visores(self):
        if self.imagen_mostrada is None: return
        
        # Si hay historial, mostramos ambos. Si no, solo el de trabajo.
        if not self.gestor.historial:
            self.visor_izq.hide()
            self.mostrar_en_label(self.visor_der, self.imagen_mostrada)
        else:
            self.visor_izq.show()
            self.mostrar_en_label(self.visor_izq, self.imagen_original)
            self.mostrar_en_label(self.visor_der, self.imagen_mostrada)

    def mostrar_en_label(self, label, img_cv):
        if img_cv is None: return
        if len(img_cv.shape) == 2:
            h, w = img_cv.shape
            fmt = QImage.Format.Format_Grayscale8
            bytes_line = w
            q_img = QImage(img_cv.data, w, h, bytes_line, fmt)
        else:
            h, w, c = img_cv.shape
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            fmt = QImage.Format.Format_RGB888
            bytes_line = 3 * w
            q_img = QImage(img_rgb.data, w, h, bytes_line, fmt)

        pixmap = QPixmap.fromImage(q_img)
        # Escalado suave manteniendo aspecto
        if label.width() > 0 and label.height() > 0:
             label.setPixmap(pixmap.scaled(
                label.width(), label.height(), 
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

    def resizeEvent(self, event):
        # Re-escalar imágenes al cambiar tamaño de ventana
        self.actualizar_visores()
        super().resizeEvent(event)

    def mostrar_canales(self):
        if self.imagen_mostrada is None: return
        if not hasattr(self, 'modelo_actual'): self.modelo_actual = "RGB"
        canales = analisis.separar_canales(self.imagen_mostrada, self.modelo_actual)
        dialogo = VentanaCanales(canales)
        dialogo.exec()

    def mostrar_componentes(self):
        if self.imagen_mostrada is None: return
        img_etiquetada, cantidad = analisis.etiquetar_componentes(self.imagen_mostrada)
        QMessageBox.information(self, "Análisis", f"Se encontraron {cantidad} objetos.")
        dialogo = VentanaCanales([("Mapa de Etiquetas", img_etiquetada)])
        dialogo.setWindowTitle(f"Componentes Conexas ({cantidad} objetos)")
        dialogo.exec()

    def aplicar_modelo(self, modelo):
        if self.imagen_mostrada is None: return
        self.gestor.guardar_estado(self.imagen_mostrada)
        self.modelo_actual = modelo 
        resultado = colores.aplicar_modelo(self.imagen_mostrada, modelo)
        self.imagen_mostrada = resultado
        self.actualizar_visores()
        self.lbl_info.setText(f"Modelo aplicado: {modelo}")

    def aplicar_morfologia(self, operacion):
        if self.imagen_mostrada is None: return
        self.gestor.guardar_estado(self.imagen_mostrada)
        KERNEL_FIJO = 5  
        resultado = None
        if operacion == "Erosión": resultado = morfologia.erosion(self.imagen_mostrada, KERNEL_FIJO)
        elif operacion == "Dilatación": resultado = morfologia.dilatacion(self.imagen_mostrada, KERNEL_FIJO)
        elif operacion == "Apertura": resultado = morfologia.apertura_ex(self.imagen_mostrada, KERNEL_FIJO)
        elif operacion == "Cierre": resultado = morfologia.cierre_ex(self.imagen_mostrada, KERNEL_FIJO)

        if resultado is not None:
            self.imagen_mostrada = resultado
            self.modelo_actual = "GRAY"
            self.actualizar_visores()
            self.lbl_info.setText(f"Morfología: {operacion}")

    def aplicar_filtro(self, nombre_filtro):
        if self.imagen_mostrada is None: return
        self.gestor.guardar_estado(self.imagen_mostrada)
        KERNEL_STD = 3
        resultado = None
        try:
            if nombre_filtro == "Promedio": resultado = filtros.filtro_promedio(self.imagen_mostrada, KERNEL_STD)
            elif nombre_filtro == "Mediana": resultado = filtros.filtro_mediana(self.imagen_mostrada, KERNEL_STD)
            elif nombre_filtro == "Gaussiano": resultado = filtros.filtro_gaussiano(self.imagen_mostrada, KERNEL_STD)
            elif nombre_filtro == "Máximo": resultado = filtros.filtro_maximo(self.imagen_mostrada, KERNEL_STD)
            elif nombre_filtro == "Mínimo": resultado = filtros.filtro_minimo(self.imagen_mostrada, KERNEL_STD)
            elif nombre_filtro == "Sobel": resultado = filtros.filtro_sobel(self.imagen_mostrada)
            elif nombre_filtro == "Prewitt": resultado = filtros.filtro_prewitt(self.imagen_mostrada)
            elif nombre_filtro == "Canny": resultado = filtros.filtro_canny(self.imagen_mostrada)
            elif nombre_filtro == "Laplaciano": resultado = filtros.filtro_laplaciano(self.imagen_mostrada)

            if resultado is not None:
                self.imagen_mostrada = resultado
                if len(resultado.shape) == 2: self.modelo_actual = "GRAY"
                self.actualizar_visores()
                self.lbl_info.setText(f"Filtro: {nombre_filtro}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
    
    def aplicar_logica(self, tipo_operacion):
        if self.imagen_mostrada is None: return
        if tipo_operacion == "NOT":
            self.gestor.guardar_estado(self.imagen_mostrada)
            self.imagen_mostrada = operaciones_logicas.operacion_not(self.imagen_mostrada)
            self.actualizar_visores()
            self.lbl_info.setText("Lógica: NOT")
            return
        archivo, _ = QFileDialog.getOpenFileName(self, f"Imagen para {tipo_operacion}", "", "Img (*.png *.jpg *.bmp)")
        if not archivo: return
        img_sec = cv2.imread(archivo)
        if img_sec is None: return
        self.gestor.guardar_estado(self.imagen_mostrada)
        res = None
        if tipo_operacion == "AND": res = operaciones_logicas.operacion_and(self.imagen_mostrada, img_sec)
        elif tipo_operacion == "OR": res = operaciones_logicas.operacion_or(self.imagen_mostrada, img_sec)
        elif tipo_operacion == "XOR": res = operaciones_logicas.operacion_xor(self.imagen_mostrada, img_sec)
        
        if res is not None:
            self.imagen_mostrada = res
            self.actualizar_visores()
            self.lbl_info.setText(f"Lógica: {tipo_operacion}")

    def gestionar_aritmetica(self, operacion):
        if self.imagen_mostrada is None: return
        if operacion == "INV":
            self.gestor.guardar_estado(self.imagen_mostrada)
            self.imagen_mostrada = operaciones_aritmeticas.inversion_aritmetica(self.imagen_mostrada)
            self.actualizar_visores()
            self.lbl_info.setText("Invertido")
            return
        
        val, ok = QInputDialog.getDouble(self, "Valor", "Introduce valor (ej. 50 suma, 1.5 mult):", 1.0, -255, 255, 2)
        if not ok: return
        self.gestor.guardar_estado(self.imagen_mostrada)
        res = None
        if operacion == "SUMA": res = operaciones_aritmeticas.suma_escalar(self.imagen_mostrada, val)
        elif operacion == "MULT": res = operaciones_aritmeticas.multiplicacion_escalar(self.imagen_mostrada, val)
        elif operacion == "RESTA": res = operaciones_aritmeticas.resta_escalar(self.imagen_mostrada, val)
        
        if res is not None:
            self.imagen_mostrada = res
            self.actualizar_visores()
            self.lbl_info.setText(f"Aritmética: {operacion} ({val})")