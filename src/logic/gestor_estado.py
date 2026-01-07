class GestorEstado:
    def __init__(self):
        self.historial = []  # Pila de imágenes pasadas
        self.rehacer_stack = [] # Pila para "Adelante"
        self.max_pasos = 20  # Límite para no llenar la RAM

    def guardar_estado(self, imagen_nueva):
        """Llama a esto ANTES de modificar la imagen actual"""
        if imagen_nueva is None: return
        
        self.historial.append(imagen_nueva.copy())
        if len(self.historial) > self.max_pasos:
            self.historial.pop(0) # Borra el más viejo
        
        self.rehacer_stack.clear() # Al hacer algo nuevo, se borra el futuro

    def deshacer(self, imagen_actual):
        """Retorna la imagen anterior y guarda la actual en rehacer"""
        if not self.historial:
            return None
        
        # Guardamos la actual en rehacer por si queremos volver
        if imagen_actual is not None:
            self.rehacer_stack.append(imagen_actual)

        return self.historial.pop()

    def rehacer(self, imagen_actual):
        """Retorna la imagen siguiente"""
        if not self.rehacer_stack:
            return None
        
        # Guardamos la actual en historial
        if imagen_actual is not None:
            self.historial.append(imagen_actual)
            
        return self.rehacer_stack.pop()

    def reiniciar(self):
        self.historial.clear()
        self.rehacer_stack.clear()
