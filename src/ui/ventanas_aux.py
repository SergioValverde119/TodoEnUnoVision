from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QFormLayout
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import cv2

class VentanaHistograma(QDialog):
    def __init__(self, figura_matplotlib, estadisticas=None):
        super().__init__()
        self.setWindowTitle("Histograma y Propiedades")
        self.resize(800, 500)
        
        # Layout Principal Horizontal: Gráfico a la izq, Datos a la der
        layout_main = QHBoxLayout()
        
        # 1. Canvas Matplotlib
        canvas = FigureCanvas(figura_matplotlib)
        layout_main.addWidget(canvas, stretch=2) # El gráfico ocupa más espacio
        
        # 2. Panel de Estadísticas (Si se proveen)
        if estadisticas:
            group_box = QGroupBox("Propiedades Estadísticas")
            form_layout = QFormLayout()
            
            # Estilo para los textos
            estilo_valor = "font-weight: bold; color: #0078d7; font-size: 12px;"
            
            for clave, valor in estadisticas.items():
                lbl_val = QLabel(str(valor))
                lbl_val.setStyleSheet(estilo_valor)
                form_layout.addRow(f"{clave}:", lbl_val)
            
            group_box.setLayout(form_layout)
            
            # Contenedor vertical derecho para alinear arriba
            layout_derecha = QVBoxLayout()
            layout_derecha.addWidget(group_box)
            layout_derecha.addStretch() # Empuja todo hacia arriba
            
            layout_main.addLayout(layout_derecha, stretch=1)

        self.setLayout(layout_main)

class VentanaCanales(QDialog):
    def __init__(self, lista_canales):
        super().__init__()
        self.setWindowTitle("Canales Separados")
        self.resize(900, 400)
        layout = QHBoxLayout()
        
        for nombre, img in lista_canales:
            v_box = QVBoxLayout()
            lbl_titulo = QLabel(nombre)
            lbl_titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_titulo.setStyleSheet("font-weight: bold; font-size: 14px;")
            
            lbl_img = QLabel()
            lbl_img.setScaledContents(True)
            lbl_img.setFixedSize(250, 250)
            lbl_img.setStyleSheet("border: 1px solid gray;")
            
            if len(img.shape) == 2:
                h, w = img.shape
                fmt = QImage.Format.Format_Grayscale8
            else:
                h, w, c = img.shape
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                fmt = QImage.Format.Format_RGB888
            
            q_img = QImage(img.data, w, h, w*3 if len(img.shape)>2 else w, fmt)
            lbl_img.setPixmap(QPixmap.fromImage(q_img))
            
            v_box.addWidget(lbl_titulo)
            v_box.addWidget(lbl_img)
            layout.addLayout(v_box)
            
        self.setLayout(layout)