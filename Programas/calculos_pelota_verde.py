import cv2
import numpy as np

# Variables globales para controlar el flujo del video
pausar_video = False       # True = el video se pausa
frame_a_frame = False      # True = avanzar manualmente un frame

# Función de callback del mouse
def mouse_callback(event, x, y, flags, param):
    global pausar_video, frame_a_frame

    if event == cv2.EVENT_RBUTTONDOWN:  # Click derecho -> Pausar/Reanudar
        pausar_video = not pausar_video

    if event == cv2.EVENT_LBUTTONDOWN:  # Click izquierdo -> Avanzar 1 frame
        if pausar_video:
            frame_a_frame = True


# Parámetros de entrada
video_path = "V1_procesado.mp4"     # Ruta del video
diametro_real_cm = 9           # Diámetro real de la pelota (en cm)

# Abrir el video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: no se pudo abrir el video")
    exit()

# Obtener FPS del video (para calcular tiempos)
fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1.0 / fps  # duración de cada frame en segundos

# Rango de color en HSV para detectar la pelota verde
lower_green = np.array([35, 100, 50])
upper_green = np.array([85, 255, 255])

# Listas para guardar datos de análisis
tiempos = []      # tiempo asociado a cada frame
centros = []      # posiciones (x,y) del centro de la pelota

frame_idx = 0
escala_cm_px = None  # Escala de conversión de píxeles a cm

# Crear ventana y asociar callback del mouse
cv2.namedWindow("Video")
cv2.setMouseCallback("Video", mouse_callback)

# Bucle principal de procesamiento de video
while True:
    # Leer un nuevo frame SOLO si no está en pausa o si se pidió avanzar 1 frame
    if not pausar_video or frame_a_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar el frame y Convertir a HSV
        frame = cv2.resize(frame, (540, 960))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Crear máscara binaria para detectar el color verde
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Filtrar ruido con operaciones morfológicas (apertura + cierre)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # MORPH_OPEN (Apertura) = Erosión + Dilatación
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # MORPH_CLOSE (Cierre) = Dilatación + Erosión

        # Buscar contornos en la máscara
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contorno in contornos:
            area = cv2.contourArea(contorno)

            # Ignorar objetos muy pequeños (ruido)
            if area > 500:
                # Obtener círculo mínimo que encierra el contorno
                (cx, cy), radio = cv2.minEnclosingCircle(contorno)
                centro = (int(cx), int(cy))
                radio = int(radio) # círculo 10% más pequeño

                # Calcular escala (cm/px) usando el diámetro real de la pelota
                if escala_cm_px is None:
                    diametro_px = 2 * radio
                    escala_cm_px = diametro_real_cm / diametro_px

                # Guardar datos de análisis
                tiempos.append(frame_idx * dt)          # tiempo en segundos
                centros.append((centro[0], centro[1]))  # posición en píxeles

                # Dibujar pelota detectada
                cv2.circle(frame, centro, radio, (0, 255, 0), 4)    # círculo verde
                cv2.circle(frame, centro, 3, (0, 0, 255), -1)          # centro rojo

        frame_idx += 1
        frame_a_frame = False  # Resetear avance manual después de un frame

    # Mostrar imágenes
    cv2.imshow("Video", frame)
    cv2.imshow("mascara", mask)

    # Salida con tecla ESC
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# -------------------------------------------------------------
# Análisis de movimiento (posterior a la detección)
# -------------------------------------------------------------
centros = np.array(centros, dtype=float)
tiempos = np.array(tiempos)

# Convertir coordenadas de píxeles a cm usando la escala
centros_cm = centros * escala_cm_px

# Calcular velocidades (derivada discreta de la posición)
velocidades = np.diff(centros_cm, axis=0) / dt

# Calcular aceleraciones (derivada discreta de la velocidad)
aceleraciones = np.diff(velocidades, axis=0) / dt

# Magnitudes
vel_magnitudes = np.linalg.norm(velocidades, axis=1)        # módulo de cada velocidad
acel_magnitudes = np.linalg.norm(aceleraciones, axis=1)     # módulo de cada aceleración

# ----------------- Mostrar resultados -----------------
print("Resultados del análisis de movimiento):")
print(f"Escala: 1 px = {escala_cm_px}")
print(f"Total de frames analizados: {len(centros)}")

# Velocidades por instante
print("\nVelocidades por frame (cm/s):")
for i, v in enumerate(vel_magnitudes):
    print(f"t = {tiempos[i+1]:.3f} s -> {v:.2f} cm/s")

# Aceleraciones por instante
print("\nAceleraciones por frame (cm/s²):")
for i, a in enumerate(acel_magnitudes):
    print(f"t = {tiempos[i+2]:.3f} s -> {a:.2f} cm/s²")

# Promedios globales
print("\nResumen global:")
print(f"Velocidad promedio: {np.mean(vel_magnitudes):.2f} cm/s")
print(f"Aceleracion promedio: {np.mean(acel_magnitudes):.2f} cm/s²")