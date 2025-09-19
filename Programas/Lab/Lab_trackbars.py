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


cv2.namedWindow("Ajustes Lab")
# Trackbars para el rango Lab (L, a, b)
cv2.createTrackbar("L Min", "Ajustes Lab", 42, 255, nothing)
cv2.createTrackbar("L Max", "Ajustes Lab", 121, 255, nothing)
cv2.createTrackbar("a Min", "Ajustes Lab", 90, 255, nothing)
cv2.createTrackbar("a Max", "Ajustes Lab", 118, 255, nothing)
cv2.createTrackbar("b Min", "Ajustes Lab", 150, 255, nothing)
cv2.createTrackbar("b Max", "Ajustes Lab", 255, 255, nothing)

video_path = "../Videos Procesados/V1_procesado.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: no se pudo abrir el video")
    exit()

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", mouse_callback)

while True:
    # Solo leer un nuevo frame si no está pausado
    if not pausar_video or frame_a_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (540, 960))
        ultimo_frame = frame.copy()  # Guardamos este frame
        frame_a_frame = False

    if ultimo_frame is not None:
        # Convertir a espacio Lab
        lab = cv2.cvtColor(ultimo_frame, cv2.COLOR_BGR2Lab)

        # Leer valores de trackbars
        l_min = cv2.getTrackbarPos("L Min", "Ajustes Lab")
        l_max = cv2.getTrackbarPos("L Max", "Ajustes Lab")
        a_min = cv2.getTrackbarPos("a Min", "Ajustes Lab")
        a_max = cv2.getTrackbarPos("a Max", "Ajustes Lab")
        b_min = cv2.getTrackbarPos("b Min", "Ajustes Lab")
        b_max = cv2.getTrackbarPos("b Max", "Ajustes Lab")

        lower_lab = np.array([l_min, a_min, b_min])
        upper_lab = np.array([l_max, a_max, b_max])

        # Crear máscara
        mask_lab = cv2.inRange(lab, lower_lab, upper_lab)

        # Filtrado morfológico
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((7, 7), np.uint8)
        mask_lab = cv2.morphologyEx(mask_lab, cv2.MORPH_OPEN, kernel_small)
        mask_lab = cv2.morphologyEx(mask_lab, cv2.MORPH_CLOSE, kernel_large)
        mask_lab = cv2.medianBlur(mask_lab, 5)

        # Mostrar resultados
        cv2.imshow("Video", ultimo_frame)
        cv2.imshow("Mascara Lab", mask_lab)

    if cv2.waitKey(10) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()