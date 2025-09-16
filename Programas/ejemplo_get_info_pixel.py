import cv2
import numpy as np

# Variables globales
pausar_video = False
mostrar_pixel = False

# Función de callback del mouse
def mouse_callback(event, _x, _y, flags, param):
    global pausar_video, mostrar_pixel, x, y
    if event == cv2.EVENT_LBUTTONDOWN:  # Click izquierdo -> mostrar valor HSV del píxel
        x, y = _x, _y
        mostrar_pixel = True
    if event == cv2.EVENT_RBUTTONDOWN:  # Click derecho -> pausar/reanudar video
        pausar_video = not pausar_video
        
# Cargar el video
video_path = 'V1_procesado.mp4'  # Cambia esto al camino de tu video
cap = cv2.VideoCapture(video_path, cv2.CAP_MSMF)

#validate if cap is open
if not cap.isOpened():
    print("Error: no se pudo abrir video")
else:
    print("video abierto correctamente")
# Ventanas
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', mouse_callback)
cv2.namedWindow("Trackbars")

# Dimensiones fijas
nuevo_alto = 960
nuevo_ancho = 540

upper_limit = np.array([225,120,35])
lower_limit = np.array([221,118,31])


def nothing(x):
    pass

# Crear una ventana con controles deslizantes (trackbars)
cv2.namedWindow("Trackbars")
cv2.createTrackbar("H Min", "Trackbars", 35, 179, nothing)
cv2.createTrackbar("H Max", "Trackbars", 85, 179, nothing)
cv2.createTrackbar("S Min", "Trackbars", 100, 255, nothing)
cv2.createTrackbar("S Max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("V Min", "Trackbars", 30, 255, nothing)
cv2.createTrackbar("V Max", "Trackbars", 255, 255, nothing)


while True:
    if not pausar_video:
        ret, frame = cap.read()
        #frame2 = np.zeros(frame.shape, dtype=np.uint8)
        frame2 = np.copy(frame)
        if not ret:
            break
    if mostrar_pixel:
        pixel_color = frame[y, x]  # Obtener el valor del color en (x, y)
        frame2 = np.copy(frame)
        cv2.putText(frame2, f'Posición (x, y): ({x}, {y}) Color: {pixel_color}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)
    
    frame3 = cv2.resize(frame2, (nuevo_ancho, nuevo_alto)) 
    # convertir bgr a hsv
    hsv = cv2.cvtColor(frame3, cv2.COLOR_BGR2HSV)
    
    # Leer valores de las trackbars
    h_min = cv2.getTrackbarPos("H Min", "Trackbars")
    h_max = cv2.getTrackbarPos("H Max", "Trackbars")
    s_min = cv2.getTrackbarPos("S Min", "Trackbars")
    s_max = cv2.getTrackbarPos("S Max", "Trackbars")
    v_min = cv2.getTrackbarPos("V Min", "Trackbars")
    v_max = cv2.getTrackbarPos("V Max", "Trackbars")

    # Crear los rangos de colores
    bajo = np.array([h_min, s_min, v_min])
    alto = np.array([h_max, s_max, v_max])
    ## binarizar imagen
    mascara = cv2.inRange(hsv, bajo, alto)

    # Filtrar ruido
    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)



    cv2.imshow('Video mascara', mascara)
    cv2.imshow('Video', hsv)
    
    key = cv2.waitKey(33) & 0xFF
    if key == 27:  # Presiona la tecla 'Esc' para salir
        break
cap.release()
cv2.destroyAllWindows()