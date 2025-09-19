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

# Crear ventana de trackbars
cv2.namedWindow("Ajustes YCrCb")
cv2.createTrackbar("Y Min", "Ajustes YCrCb", 60, 255, nothing)
cv2.createTrackbar("Y Max", "Ajustes YCrCb", 255, 255, nothing)
cv2.createTrackbar("Cr Min", "Ajustes YCrCb", 50, 255, nothing)
cv2.createTrackbar("Cr Max", "Ajustes YCrCb", 102, 255, nothing)
cv2.createTrackbar("Cb Min", "Ajustes YCrCb", 70, 255, nothing)
cv2.createTrackbar("Cb Max", "Ajustes YCrCb", 170, 255, nothing)

video_path = "../Videos Procesados/R2_procesado.mp4"
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
        # Convertir a espacio YCrCb
        ycrcb = cv2.cvtColor(ultimo_frame, cv2.COLOR_BGR2YCrCb)
        ycrcb = cv2.bitwise_not(ycrcb)  # Invertir: ahora la pelota es blanca y el fondo negro

        # Leer valores de trackbars
        y_min = cv2.getTrackbarPos("Y Min", "Ajustes YCrCb")
        y_max = cv2.getTrackbarPos("Y Max", "Ajustes YCrCb")
        cr_min = cv2.getTrackbarPos("Cr Min", "Ajustes YCrCb")
        cr_max = cv2.getTrackbarPos("Cr Max", "Ajustes YCrCb")
        cb_min = cv2.getTrackbarPos("Cb Min", "Ajustes YCrCb")
        cb_max = cv2.getTrackbarPos("Cb Max", "Ajustes YCrCb")

        lower_ycrcb = np.array([y_min, cr_min, cb_min])
        upper_ycrcb = np.array([y_max, cr_max, cb_max])

        # Crear máscara
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

        # Filtrado morfológico
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((7, 7), np.uint8)
        mask_ycrcb = cv2.morphologyEx(mask_ycrcb, cv2.MORPH_OPEN, kernel_small)
        mask_ycrcb = cv2.morphologyEx(mask_ycrcb, cv2.MORPH_CLOSE, kernel_large)
        mask_ycrcb = cv2.medianBlur(mask_ycrcb, 5)

        # Mostrar resultados
        cv2.imshow("Video", ultimo_frame)
        cv2.imshow("Mascara YCrCb", mask_ycrcb)

    if cv2.waitKey(10) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
