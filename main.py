import sys
from PyQt6.QtWidgets import QApplication
from src.ui.ventana import VentanaPrincipal

if __name__ == "__main__":
    # 1. Crea la instancia de la aplicación
    app = QApplication(sys.argv)
    
    # 2. Crea la ventana que diseñamos
    ventana = VentanaPrincipal()
    
    # 3. Muestra la ventana en pantalla
    ventana.show()
    
    # 4. Inicia el bucle de eventos
    sys.exit(app.exec())
