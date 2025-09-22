import cv2
import numpy as np

pausar_video = False
frame_a_frame = False
ultimo_frame = None  # Guardar el último frame para reprocesarlo cuando se pausa

def mouse_callback(event, x, y, flags, param):
    global pausar_video, frame_a_frame
    if event == cv2.EVENT_RBUTTONDOWN:  # Pausar/reanudar con clic derecho
        pausar_video = not pausar_video
    if event == cv2.EVENT_LBUTTONDOWN:  # Avanzar un frame cuando está pausado
        if pausar_video:
            frame_a_frame = True

def nothing(x):
    pass

# Ventana y trackbars para HSV
cv2.namedWindow("Ajustes HSV")
cv2.createTrackbar("H Min", "Ajustes HSV", 5, 179, nothing)
cv2.createTrackbar("H Max", "Ajustes HSV", 168, 179, nothing)
cv2.createTrackbar("S Min", "Ajustes HSV", 50, 255, nothing)
cv2.createTrackbar("S Max", "Ajustes HSV", 255, 255, nothing)
cv2.createTrackbar("V Min", "Ajustes HSV", 60, 255, nothing)
cv2.createTrackbar("V Max", "Ajustes HSV", 255, 255, nothing)

video_path = "../Videos Procesados/V1_procesado.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: no se pudo abrir el video")
    exit()

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", mouse_callback)

while True:
    if not pausar_video or frame_a_frame:
        ret, frame = cap.read()
        if not ret:
            break
        #frame = cv2.resize(frame, (540, 960))   #desktop
        frame = cv2.resize(frame, (360, 640))  #laptop
        ultimo_frame = frame.copy()
        frame_a_frame = False

    if ultimo_frame is not None:
        # Convertir a espacio HSV
        hsv = cv2.cvtColor(ultimo_frame, cv2.COLOR_BGR2HSV)

        # Leer valores de trackbars
        h_min = cv2.getTrackbarPos("H Min", "Ajustes HSV")
        h_max = cv2.getTrackbarPos("H Max", "Ajustes HSV")
        s_min = cv2.getTrackbarPos("S Min", "Ajustes HSV")
        s_max = cv2.getTrackbarPos("S Max", "Ajustes HSV")
        v_min = cv2.getTrackbarPos("V Min", "Ajustes HSV")
        v_max = cv2.getTrackbarPos("V Max", "Ajustes HSV")

        lower_hsv = np.array([h_min, s_min, v_min])
        upper_hsv = np.array([h_max, s_max, v_max])

        # Crear máscara
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # Filtrado morfológico
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((7, 7), np.uint8)
        mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel_small)
        mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel_large)
        mask_hsv = cv2.medianBlur(mask_hsv, 5)

        # Mostrar resultados
        cv2.imshow("Video", ultimo_frame)
        cv2.imshow("Mascara HSV", mask_hsv)

    if cv2.waitKey(10) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
