import cv2
import os
from datetime import datetime
from PyQt6.QtWidgets import (
    QMainWindow, QLabel, QScrollArea, QPushButton, QVBoxLayout, QHBoxLayout, 
    QWidget, QFileDialog, QMessageBox, QMenu, QSizePolicy, QInputDialog
)
from PyQt6.QtGui import QImage, QPixmap, QAction
from PyQt6.QtCore import Qt

# Importamos tus módulos de lógica
from src.logic.gestor_estado import GestorEstado
from src.ui.ventanas_aux import VentanaHistograma, VentanaCanales
from src.logic import analisis
from src.logic import operaciones_aritmeticas
from src.logic import operaciones_logicas
from src.logic import colores
from src.logic import mapas
from src.logic import filtros
from src.logic import morfologia

class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Analysis App")
        self.resize(900, 700)

        # Rutas y Gestor
        base_dir = os.getcwd()
        self.ruta_data = os.path.join(base_dir, "data")
        self.ruta_salidas = os.path.join(base_dir, "salidas")
        os.makedirs(self.ruta_salidas, exist_ok=True)
        
        self.gestor = GestorEstado()
        
        # Variables de estado
        self.imagen_original = None
        self.imagen_mostrada = None

        # 1. Crear Menús (Barra superior)
        self.crear_menus()

        # 2. Interfaz Central (Visor)
        self.init_ui()

    def init_ui(self):
        """Configura los visores (Original vs Resultado)"""
        # Usamos un layout horizontal para poner las imágenes lado a lado
        self.layout_visores = QHBoxLayout()
        
        # --- Visor Izquierdo (Original / Anterior) ---
        self.visor_izq = QLabel("Original")
        self.visor_izq.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.visor_izq.setStyleSheet("border: 1px dashed #555;")
        self.visor_izq.setScaledContents(False) # Importante para evitar efecto lupa
        
        # --- Visor Derecho (Resultado) ---
        self.visor_der = QLabel("Resultado")
        self.visor_der.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.visor_der.setStyleSheet("border: 2px solid #0078d7;") # Borde azul para destacar
        self.visor_der.setScaledContents(False)

        # Política de tamaño para evitar que exploten
        from PyQt6.QtWidgets import QSizePolicy
        policy = QSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.visor_izq.setSizePolicy(policy)
        self.visor_der.setSizePolicy(policy)

        # Agregamos al layout
        self.layout_visores.addWidget(self.visor_izq)
        self.layout_visores.addWidget(self.visor_der)
        
        # Por defecto, ocultamos el izquierdo hasta que haya un cambio
        self.visor_izq.hide()

        # Widget contenedor
        widget_central = QWidget()
        widget_central.setLayout(self.layout_visores)
        self.setCentralWidget(widget_central)

    def crear_menus(self):
        barra_menu = self.menuBar()
        
        # 1. ARCHIVO
        menu_archivo = barra_menu.addMenu("Archivo")
        menu_archivo.addAction("Cargar Imagen", self.cargar_imagen)
        menu_archivo.addAction("Guardar Imagen (PNG)", self.guardar_imagen)
        
        # 2. EDICIÓN (Undo/Redo)
        menu_edicion = barra_menu.addMenu("Edición")
        menu_edicion.addAction("Atrás", self.accion_atras)
        menu_edicion.addAction("Adelante", self.accion_adelante)
        menu_edicion.addSeparator()
        menu_edicion.addAction("Restablecer", self.accion_restablecer)

        # 3. VER (Histograma y Canales)
        menu_ver = barra_menu.addMenu("Ver")
        menu_ver.addAction("Mostrar Histograma", self.mostrar_histograma) 
        menu_ver.addAction("Mostrar Canales", self.mostrar_canales)
        menu_ver.addAction("Componentes Conexas", self.mostrar_componentes)

        # 4. MODELOS DE COLOR
        menu_color = barra_menu.addMenu("Modelos Color")
        menu_color.addAction("RGB", lambda: self.aplicar_modelo("RGB"))
        menu_color.addAction("Escala de Grises", lambda: self.aplicar_modelo("GRAY"))
        menu_color.addAction("Binarizar", lambda: self.aplicar_modelo("BINARY"))
        menu_color.addAction("HSV", lambda: self.aplicar_modelo("HSV"))
        menu_color.addAction("CMYK", lambda: self.aplicar_modelo("CMYK"))

        # 5. ARITMÉTICAS
        menu_arit = barra_menu.addMenu("Aritméticas")
        menu_arit.addAction("Suma", lambda: self.gestionar_aritmetica("SUMA"))
        menu_arit.addAction("Resta", lambda: self.gestionar_aritmetica("RESTA"))
        menu_arit.addAction("Multiplicación", lambda: self.gestionar_aritmetica("MULT"))
        menu_arit.addAction("División", lambda: self.gestionar_aritmetica("DIV"))
        menu_arit.addSeparator()
        menu_arit.addAction("Inversión", lambda: self.gestionar_aritmetica("INV"))

        # 6. OPERACIONES LÓGICAS
        menu_logicas = barra_menu.addMenu("Lógicas")
        menu_logicas.addAction("NOT - Invertir", lambda: self.aplicar_logica("NOT"))
        menu_logicas.addSeparator()
        # Estas requieren una segunda imagen
        menu_logicas.addAction("AND - Intersección", lambda: self.aplicar_logica("AND"))
        menu_logicas.addAction("OR - Unión", lambda: self.aplicar_logica("OR"))
        menu_logicas.addAction("XOR", lambda: self.aplicar_logica("XOR"))

        # 7. MAPAS DE COLOR
        menu_mapas = barra_menu.addMenu("Mapas Color")
        menu_mapas.addAction("JET", lambda: self.aplicar_mapa_color("JET"))
        menu_mapas.addAction("HOT", lambda: self.aplicar_mapa_color("HOT"))
        menu_mapas.addAction("OCEAN", lambda: self.aplicar_mapa_color("OCEAN"))
        menu_mapas.addAction("BONE", lambda: self.aplicar_mapa_color("BONE"))
        menu_mapas.addAction("PINK", lambda: self.aplicar_mapa_color("PINK"))
        menu_mapas.addAction("PROPIO 1", lambda: self.aplicar_mapa_color("PROPIO 1"))
        menu_mapas.addAction("PROPIO 2", lambda: self.aplicar_mapa_color("PROPIO 2"))

        # 8. MORFOLOGÍAS
        menu_morfo = barra_menu.addMenu("Morfologías")
        # Operaciones Directas
        menu_morfo.addAction("Erosión", lambda: self.aplicar_morfologia("Erosión"))
        menu_morfo.addAction("Dilatación", lambda: self.aplicar_morfologia("Dilatación"))
        menu_morfo.addSeparator()
        # Aperturas
        menu_morfo.addAction("Apertura", lambda: self.aplicar_morfologia("Apertura"))
        menu_morfo.addAction("Apertura EX", lambda: self.aplicar_morfologia("Apertura EX"))
        menu_morfo.addSeparator()
        # Cierres
        menu_morfo.addAction("Cierre", lambda: self.aplicar_morfologia("Cierre"))
        menu_morfo.addAction("Cierre EX", lambda: self.aplicar_morfologia("Cierre EX"))

        # 9. FILTRADO
        menu_filtros = barra_menu.addMenu("Filtros")
        # Submenú Ruido
        menu_ruido = menu_filtros.addMenu("Reducción de Ruido")
        menu_ruido.addAction("Promedio", lambda: self.aplicar_filtro("Promedio"))
        menu_ruido.addAction("Mediana", lambda: self.aplicar_filtro("Mediana"))
        menu_ruido.addAction("Gaussiano", lambda: self.aplicar_filtro("Gaussiano"))
        menu_ruido.addAction("Máximo", lambda: self.aplicar_filtro("Máximo"))
        menu_ruido.addAction("Mínimo", lambda: self.aplicar_filtro("Mínimo"))
        menu_filtros.addSeparator()
        # Submenú Bordes
        menu_bordes = menu_filtros.addMenu("Detección de Bordes")
        menu_bordes.addAction("Sobel", lambda: self.aplicar_filtro("Sobel"))
        menu_bordes.addAction("Prewitt", lambda: self.aplicar_filtro("Prewitt"))
        menu_bordes.addAction("Roberts", lambda: self.aplicar_filtro("Roberts"))
        menu_bordes.addAction("Canny", lambda: self.aplicar_filtro("Canny"))
        menu_bordes.addAction("Laplaciano", lambda: self.aplicar_filtro("Laplaciano"))
        menu_bordes.addAction("Kirsch", lambda: self.aplicar_filtro("Kirsch"))


    # ==========================================
    #              FUNCIONES LÓGICAS 
    # ==========================================

    def cargar_imagen(self):
        archivo, _ = QFileDialog.getOpenFileName(self, "Abrir imagen", self.ruta_data, "Imagenes (*.png *.jpg *.bmp *.tif)")
        if archivo:
            img = cv2.imread(archivo)
            if img is None:
                QMessageBox.critical(self, "Error", "No se pudo leer la imagen.")
                return
            
            # Reset total
            self.gestor.reiniciar()
            self.modelo_actual = "RGB"  # Reset del modelo de color
            
            # Guardamos la original limpia
            self.imagen_original = img
            # La mostrada inicial es una copia de la original
            self.imagen_mostrada = img.copy()
            
            # Actualizamos interfaz
            self.actualizar_visores()

    def guardar_imagen(self):
        if self.imagen_mostrada is None:
            QMessageBox.warning(self, "Aviso", "No hay imagen para guardar.")
            return

        nombre_archivo = f"resultado_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        ruta_completa = os.path.join(self.ruta_salidas, nombre_archivo)
        
        cv2.imwrite(ruta_completa, self.imagen_mostrada)
        QMessageBox.information(self, "Guardado", f"Imagen guardada en:\n{ruta_completa}")

    def accion_atras(self):
        imagen_anterior = self.gestor.deshacer(self.imagen_mostrada)
        
        if imagen_anterior is not None:
            self.imagen_mostrada = imagen_anterior
            # Reseteamos el modelo a RGB por seguridad para evitar conflictos con histogramas
            self.modelo_actual = "RGB"
            # Actualizamos la interfaz de doble visor
            self.actualizar_visores() 

    def accion_adelante(self):
        imagen_siguiente = self.gestor.rehacer(self.imagen_mostrada)
        
        if imagen_siguiente is not None:
            self.imagen_mostrada = imagen_siguiente
            self.modelo_actual = "RGB"
            self.actualizar_visores()

    def accion_restablecer(self):
        if self.imagen_original is not None:
            self.gestor.reiniciar()
            self.imagen_mostrada = self.imagen_original.copy()
            self.modelo_actual = "RGB"
            self.actualizar_visores()


    def actualizar_visores(self):
        """Muestra las imágenes en los visores correspondientes"""
        if self.imagen_mostrada is None:
            return

        # Lógica: 
        # Si NO hay historial (es la primera carga), solo mostramos Resultado (derecha) y ocultamos Original (izq).
        # Si SÍ hay historial (ya editamos algo), mostramos ambas.
        
        if not self.gestor.historial:
            # Caso inicial: Solo una imagen
            self.visor_izq.hide()
            self.mostrar_en_label(self.visor_der, self.imagen_mostrada)
        else:
            # Caso edición: Mostramos comparación
            self.visor_izq.show()
            self.mostrar_en_label(self.visor_izq, self.imagen_original) # Siempre mostramos la original base a la izquierda
            self.mostrar_en_label(self.visor_der, self.imagen_mostrada)

    def mostrar_en_label(self, label, img_cv):
        """Convierte OpenCV -> Qt y pone la imagen en el label"""
        if img_cv is None: return
        
        # Detección de formato para conversión
        if len(img_cv.shape) == 2: # Grises/Binario
            h, w = img_cv.shape
            fmt = QImage.Format.Format_Grayscale8
            bytes_line = w
            q_img = QImage(img_cv.data, w, h, bytes_line, fmt)
            
        else: # Color
            h, w, c = img_cv.shape
            # Convertimos BGR a RGB para visualizar correctamente
            # PERO si es una simulación CMYK o HSV rara, igual la mostramos como RGB
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            fmt = QImage.Format.Format_RGB888
            bytes_line = 3 * w
            q_img = QImage(img_rgb.data, w, h, bytes_line, fmt)

        pixmap = QPixmap.fromImage(q_img)
        
        # Escalado suave
        label.setPixmap(pixmap.scaled(
            label.width(), label.height(), 
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    # ==========================================
    #           FUNCIONES DE ANÁLISIS
    # ==========================================

    def mostrar_histograma(self):
        if self.imagen_mostrada is None:
            QMessageBox.warning(self, "Aviso", "Primero carga una imagen.")
            return
            
        # Nos aseguramos de tener un modelo definido, si no, RGB por defecto
        if not hasattr(self, 'modelo_actual'):
            self.modelo_actual = "RGB"

        # 1. Calculamos la gráfica pasando el MODELO ACTUAL
        fig = analisis.calcular_histograma(self.imagen_mostrada, self.modelo_actual)
        
        # 2. Abrimos la subventana
        dialogo = VentanaHistograma(fig)
        dialogo.exec() 

    def mostrar_canales(self):
        if self.imagen_mostrada is None:
            QMessageBox.warning(self, "Aviso", "Primero carga una imagen.")
            return

        # Validación por seguridad
        if not hasattr(self, 'modelo_actual'):
            self.modelo_actual = "RGB"
            
        # Pasamos el modelo para que sepa cómo interpretar los datos
        canales = analisis.separar_canales(self.imagen_mostrada, self.modelo_actual)
        
        dialogo = VentanaCanales(canales)
        dialogo.exec()


    def mostrar_componentes(self):
        if self.imagen_mostrada is None:
            QMessageBox.warning(self, "Aviso", "Primero carga una imagen.")
            return

        # Llamamos a la lógica
        img_etiquetada, cantidad = analisis.etiquetar_componentes(self.imagen_mostrada)
        
        QMessageBox.information(self, "Análisis", f"Se encontraron {cantidad} objetos (componentes conexas).")
        
        # Opción B: Abrir en ventana aparte
        # Nota: Asegúrate de tener importada VentanaCanales al inicio del archivo
        # from src.ui.ventanas_aux import VentanaCanales 
        dialogo = VentanaCanales([("Mapa de Etiquetas", img_etiquetada)])
        dialogo.setWindowTitle(f"Componentes Conexas ({cantidad} objetos)")
        dialogo.exec()

    #           TRANSFORMACIONES DE COLOR

    def aplicar_modelo(self, modelo):
        if self.imagen_mostrada is None:
            QMessageBox.warning(self, "Aviso", "Primero carga una imagen.")
            return

        self.gestor.guardar_estado(self.imagen_mostrada)
        
        # Guardamos el modelo para que el histograma sepa qué mostrar
        self.modelo_actual = modelo 
        
        resultado = colores.aplicar_modelo(self.imagen_mostrada, modelo)
        self.imagen_mostrada = resultado
        
        # Actualizamos ambos visores (Original vs Resultado)
        self.actualizar_visores()

    def aplicar_mapa_color(self, mapa):
        # 1. Validación de seguridad
        if self.imagen_mostrada is None:
            QMessageBox.warning(self, "Aviso", "Primero carga una imagen.")
            return

        self.gestor.guardar_estado(self.imagen_mostrada)
        self.modelo_actual = "RGB" 
        resultado = mapas.aplicar_mapa_color(self.imagen_mostrada, mapa)
        self.imagen_mostrada = resultado
        self.actualizar_visores()
        
    # OPERACIONES MORFOLOGICAS

    def aplicar_morfologia(self, operacion):
        if self.imagen_mostrada is None:
            QMessageBox.warning(self, "Aviso", "Carga una imagen primero.")
            return

        # 1. Guardar estado para deshacer
        self.gestor.guardar_estado(self.imagen_mostrada)
        
        # 2. DEFINIMOS EL KERNEL FIJO AQUÍ (Estándar 5x5)
        # Si quisieras cambiarlo en el futuro, solo cambias este número.
        KERNEL_FIJO = 5  

        resultado = None

        # 3. Aplicar operación
        if operacion == "Erosión":
            resultado = morfologia.erosion(self.imagen_mostrada, KERNEL_FIJO)
            
        elif operacion == "Dilatación":
            resultado = morfologia.dilatacion(self.imagen_mostrada, KERNEL_FIJO)
            
        elif operacion == "Apertura":
            resultado = morfologia.apertura_manual(self.imagen_mostrada, KERNEL_FIJO)
            
        elif operacion == "Apertura EX":
            resultado = morfologia.apertura_ex(self.imagen_mostrada, KERNEL_FIJO)
            
        elif operacion == "Cierre":
            resultado = morfologia.cierre_manual(self.imagen_mostrada, KERNEL_FIJO)
            
        elif operacion == "Cierre EX":
            resultado = morfologia.cierre_ex(self.imagen_mostrada, KERNEL_FIJO)

        # 4. Mostrar resultado
        if resultado is not None:
            self.imagen_mostrada = resultado
            self.modelo_actual = "GRAY" # Morfología siempre devuelve gris
            self.actualizar_visores()
            self.statusBar().showMessage(f"Aplicado: {operacion}")

    # FILTROS RUIDO Y BORDES
    
    def aplicar_filtro(self, nombre_filtro):
        if self.imagen_mostrada is None:
            QMessageBox.warning(self, "Aviso", "Carga una imagen primero.")
            return

        # 1. Guardar estado
        self.gestor.guardar_estado(self.imagen_mostrada)

        # 2. 
        KERNEL_STD = 3  # Tamaño estándar para filtros que lo requieran

        resultado = None

        try:
            # --- RUIDO ---
            if nombre_filtro == "Promedio":
                resultado = filtros.filtro_promedio(self.imagen_mostrada, KERNEL_STD)
            elif nombre_filtro == "Mediana":
                resultado = filtros.filtro_mediana(self.imagen_mostrada, KERNEL_STD)
            elif nombre_filtro == "Gaussiano":
                resultado = filtros.filtro_gaussiano(self.imagen_mostrada, KERNEL_STD)
            elif nombre_filtro == "Máximo":
                resultado = filtros.filtro_maximo(self.imagen_mostrada, KERNEL_STD)
            elif nombre_filtro == "Mínimo":
                resultado = filtros.filtro_minimo(self.imagen_mostrada, KERNEL_STD)
                
            # --- BORDES ---
            elif nombre_filtro == "Sobel":
                resultado = filtros.filtro_sobel(self.imagen_mostrada)
            elif nombre_filtro == "Prewitt":
                resultado = filtros.filtro_prewitt(self.imagen_mostrada)
            elif nombre_filtro == "Roberts":
                resultado = filtros.filtro_roberts(self.imagen_mostrada)
            elif nombre_filtro == "Canny":
                resultado = filtros.filtro_canny(self.imagen_mostrada)
            elif nombre_filtro == "Laplaciano":
                resultado = filtros.filtro_laplaciano(self.imagen_mostrada)
            elif nombre_filtro == "Kirsch":
                resultado = filtros.filtro_kirsch(self.imagen_mostrada)

            # 3. Mostrar
            if resultado is not None:
                self.imagen_mostrada = resultado
                # La mayoría de filtros de bordes devuelven gris, aseguramos el modelo visual
                if len(resultado.shape) == 2:
                    self.modelo_actual = "GRAY"
                self.actualizar_visores()
                self.statusBar().showMessage(f"Filtro aplicado: {nombre_filtro}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al aplicar filtro {nombre_filtro}:\n{str(e)}")
    
    # operaciones logicas
    
    def aplicar_logica(self, tipo_operacion):
        if self.imagen_mostrada is None:
            QMessageBox.warning(self, "Aviso", "Carga una imagen base primero.")
            return

        # Operación NOT (Solo requiere 1 imagen)
        if tipo_operacion == "NOT":
            self.gestor.guardar_estado(self.imagen_mostrada)
            self.imagen_mostrada = operaciones_logicas.operacion_not(self.imagen_mostrada)
            self.actualizar_visores()
            self.statusBar().showMessage("Operación NOT aplicada")
            return

        # Operaciones Binarias (AND, OR, XOR) - Requieren seleccionar otra imagen
        archivo, _ = QFileDialog.getOpenFileName(
            self, 
            f"Seleccionar imagen para {tipo_operacion}", 
            "", 
            "Imágenes (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if not archivo:
            return  # El usuario canceló la selección

        # Cargar la segunda imagen
        img_secundaria = cv2.imread(archivo)
        
        if img_secundaria is None:
            QMessageBox.critical(self, "Error", "No se pudo cargar la segunda imagen.")
            return

        self.gestor.guardar_estado(self.imagen_mostrada)

        try:
            resultado = None
            if tipo_operacion == "AND":
                resultado = operaciones_logicas.operacion_and(self.imagen_mostrada, img_secundaria)
            elif tipo_operacion == "OR":
                resultado = operaciones_logicas.operacion_or(self.imagen_mostrada, img_secundaria)
            elif tipo_operacion == "XOR":
                resultado = operaciones_logicas.operacion_xor(self.imagen_mostrada, img_secundaria)
            
            if resultado is not None:
                self.imagen_mostrada = resultado
                self.actualizar_visores()
                self.statusBar().showMessage(f"Operación {tipo_operacion} aplicada con {archivo.split('/')[-1]}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al aplicar lógica:\n{str(e)}")

    def gestionar_aritmetica(self, operacion):
        if self.imagen_mostrada is None:
            QMessageBox.warning(self, "Aviso", "Carga una imagen base primero.")
            return

        # La inversión no requiere segunda imagen ni escalar variable.
        if operacion == "INV":
            from src.logic import operaciones_aritmeticas
            self.gestor.guardar_estado(self.imagen_mostrada)
            self.imagen_mostrada = operaciones_aritmeticas.inversion_aritmetica(self.imagen_mostrada)
            self.actualizar_visores()
            return

        # Preguntar al usuario el modo
        opciones = ["Con otra Imagen", "Con un valor Escalar"]
        item, ok = QInputDialog.getItem(self, "Seleccionar Modo", 
                                        f"¿Cómo deseas aplicar la {operacion}?", 
                                        opciones, 0, False)
        
        if not ok:
            return

        if item == "Con otra Imagen":
            self.aplicar_aritmetica_imagen(operacion)
        else:
            self.aplicar_aritmetica_escalar(operacion)

    def aplicar_aritmetica_imagen(self, tipo):
        archivo, _ = QFileDialog.getOpenFileName(self, "Seleccionar segunda imagen", "", "Imágenes (*.png *.jpg *.jpeg)")
        if not archivo: return
        
        img_sec = cv2.imread(archivo)
        if img_sec is None: return

        self.gestor.guardar_estado(self.imagen_mostrada)
        
        res = None
        if tipo == "SUMA": res = operaciones_aritmeticas.suma_imagenes(self.imagen_mostrada, img_sec)
        elif tipo == "RESTA": res = operaciones_aritmeticas.resta_imagenes(self.imagen_mostrada, img_sec)
        elif tipo == "MULT": res = operaciones_aritmeticas.multiplicacion_imagenes(self.imagen_mostrada, img_sec)
        elif tipo == "DIV": res = operaciones_aritmeticas.division_imagenes(self.imagen_mostrada, img_sec)

        if res is not None:
            self.imagen_mostrada = res
            self.actualizar_visores()

    def aplicar_aritmetica_escalar(self, tipo):
        # Pedir el número
        val, ok = QInputDialog.getDouble(self, "Valor Escalar", "Introduce el valor:", 1.0, 0, 1000, 2)
        if not ok: return

        from src.logic import operaciones_aritmeticas
        self.gestor.guardar_estado(self.imagen_mostrada)

        res = None
        if tipo == "SUMA": res = operaciones_aritmeticas.suma_escalar(self.imagen_mostrada, val)
        elif tipo == "RESTA": res = operaciones_aritmeticas.resta_escalar(self.imagen_mostrada, val)
        elif tipo == "MULT": res = operaciones_aritmeticas.multiplicacion_escalar(self.imagen_mostrada, val)
        elif tipo == "DIV": res = operaciones_aritmeticas.division_escalar(self.imagen_mostrada, val)

        if res is not None:
            self.imagen_mostrada = res
            self.actualizar_visores()


