from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QWidget
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import cv2

class VentanaHistograma(QDialog):
    def __init__(self, figura_matplotlib):
        super().__init__()
        self.setWindowTitle("Histograma")
        self.resize(600, 500)
        
        layout = QVBoxLayout()
        # El canvas de Matplotlib es un widget de Qt
        canvas = FigureCanvas(figura_matplotlib)
        layout.addWidget(canvas)
        self.setLayout(layout)

class VentanaCanales(QDialog):
    def __init__(self, lista_canales):
        super().__init__()
        self.setWindowTitle("Canales Separados")
        self.resize(900, 400)
        
        layout = QHBoxLayout()
        
        for nombre, img in lista_canales:
            # Contenedor vertical para Imagen + Título
            v_box = QVBoxLayout()
            
            lbl_titulo = QLabel(nombre)
            lbl_titulo.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_titulo.setStyleSheet("font-weight: bold; font-size: 14px;")
            
            lbl_img = QLabel()
            lbl_img.setScaledContents(True)
            lbl_img.setFixedSize(250, 250) # Tamaño fijo para vista previa
            lbl_img.setStyleSheet("border: 1px solid gray;")
            
            # Convertir a QPixmap
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
