import cv2
import numpy as np

def aplicar_filtro_ideal(imagen, tipo="PASA_BAJAS", radio_corte=30):
    """
    Aplica filtros en el dominio de la frecuencia usando FFT.
    """
    # 1. Convertir a grises si es necesario
    if len(imagen.shape) == 3:
        img_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = imagen

    # 2. Transformada de Fourier 2D
    dft = cv2.dft(np.float32(img_gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 3. Crear la máscara (Filtro Ideal: Círculo blanco o negro)
    filas, cols = img_gray.shape
    centro_filas, centro_cols = filas // 2, cols // 2
    
    # Máscara base
    mascara = np.zeros((filas, cols, 2), np.uint8)
    
    # Definir el círculo
    x, y = np.ogrid[:filas, :cols]
    mask_area = (x - centro_filas)**2 + (y - centro_cols)**2 <= radio_corte**2
    
    if tipo == "PASA_BAJAS":
        # Pasa Bajas: Deja pasar el centro (bajas frecuencias) -> Círculo Blanco
        mascara[mask_area] = 1
    else: 
        # Pasa Altas: Bloquea el centro -> Invertir máscara
        mascara[:, :] = 1
        mascara[mask_area] = 0

    # 4. Aplicar máscara y Transformada Inversa
    fshift = dft_shift * mascara
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

    # 5. Normalizar para visualización (0-255)
    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    
    return np.uint8(img_back)