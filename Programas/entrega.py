import cv2
import numpy as np

# Variables globales para controlar el flujo del video
pausar_video = False
frame_a_frame = False

def mouse_callback(event, x, y, flags, param):
    global pausar_video, frame_a_frame
    if event == cv2.EVENT_RBUTTONDOWN:
        pausar_video = not pausar_video
    if event == cv2.EVENT_LBUTTONDOWN:
        if pausar_video:
            frame_a_frame = True

# Parámetros de entrada
video_path = "V1_procesado.mp4"
diametro_real_cm = 9

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: no se pudo abrir el video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1.0 / fps

lower_green = np.array([35, 100, 50])
upper_green = np.array([85, 255, 255])

tiempos = []
centros = []
frame_idx = 0

# Para calcular escala más estable
diametros_px = []

cv2.namedWindow("Video")
cv2.setMouseCallback("Video", mouse_callback)

while True:
    if not pausar_video or frame_a_frame:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (540, 960))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area > 500:
                (cx, cy), radio = cv2.minEnclosingCircle(contorno)
                centro = (int(cx), int(cy))
                radio = int(radio)

                if frame_idx < 5:  # usar primeros 5 frames para promediar escala
                    diametros_px.append(2 * radio)

                tiempos.append(frame_idx * dt)
                centros.append((centro[0], centro[1]))

                cv2.circle(frame, centro, radio, (0, 255, 0), 4)
                cv2.circle(frame, centro, 3, (0, 0, 255), -1)

        frame_idx += 1
        frame_a_frame = False

    cv2.imshow("Video", frame)
    cv2.imshow("mascara", mask)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# -------------------------------------------------------------
# Análisis de movimiento
# -------------------------------------------------------------
centros = np.array(centros, dtype=float)
tiempos = np.array(tiempos)

# Escala promediada
escala_cm_px = diametro_real_cm / np.mean(diametros_px)
centros_cm = centros * escala_cm_px

# -------- Suavizar datos para reducir ruido --------
def suavizar(datos, ventana=5):
    return np.convolve(datos, np.ones(ventana)/ventana, mode='valid')

x_suav = suavizar(centros_cm[:,0])
y_suav = suavizar(centros_cm[:,1])
centros_suav = np.vstack((x_suav, y_suav)).T
tiempos_suav = tiempos[:len(centros_suav)]

# Calcular velocidades y aceleraciones
velocidades = np.diff(centros_suav, axis=0) / dt
aceleraciones = np.diff(velocidades, axis=0) / dt

vel_magnitudes = np.linalg.norm(velocidades, axis=1)
acel_magnitudes = np.linalg.norm(aceleraciones, axis=1)

print("Resultados del análisis de movimiento:")
print(f"Escala: 1 px = {escala_cm_px:.3f} cm")
print(f"Total de frames analizados: {len(centros)}")

print("\nVelocidades por frame (cm/s):")
for i, v in enumerate(vel_magnitudes):
    print(f"t = {tiempos_suav[i+1]:.3f} s -> {v:.2f} cm/s")

print("\nAceleraciones por frame (cm/s²):")
for i, a in enumerate(acel_magnitudes):
    print(f"t = {tiempos_suav[i+2]:.3f} s -> {a:.2f} cm/s²")

print("\nResumen global:")
print(f"Velocidad promedio: {np.mean(vel_magnitudes):.2f} cm/s")
print(f"Aceleracion promedio: {np.mean(acel_magnitudes):.2f} cm/s²")

