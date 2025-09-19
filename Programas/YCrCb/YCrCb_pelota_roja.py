import cv2
import numpy as np
import matplotlib.pyplot as plt

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
video_path = "../Videos Procesados/R2_procesado.mp4"  # Ruta del video
diametro_real_cm = 6           # Diámetro real de la pelota (en cm)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: no se pudo abrir el video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1.0 / fps

# Rango de color en YCrCb para detectar la pelota roja
lower_red = np.array([60, 50, 70])
upper_red = np.array([255, 102, 170])

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

        # Redimensionamos video a tamaño fijo
        frame = cv2.resize(frame, (540, 960))

        # Convertir a espacio YCrCb y aplicar máscara por rango de color
        YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        mask = cv2.bitwise_not(YCrCb)  # Invertir: ahora la pelota es blanca y el fondo negro
        mask = cv2.inRange(mask, lower_red, upper_red)

        # Procesamiento morfológico para limpiar la máscara
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
        mask = cv2.medianBlur(mask, 5)

        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area > 500:
                (cx, cy), radio = cv2.minEnclosingCircle(contorno)
                centro = (int(cx), int(cy))
                radio = int(radio)

                if frame_idx < 5:  # usar primeros frames para promediar escala
                    diametros_px.append(2 * radio)

                tiempos.append(frame_idx * dt)
                centros.append((centro[0], centro[1]))

                # Dibujar marcador
                cv2.circle(frame, centro, radio, (0, 255, 0), 2)
                cv2.circle(frame, centro, 3, (0, 0, 255), -1)

        frame_idx += 1
        frame_a_frame = False

    cv2.imshow("Video", frame)
    cv2.imshow("Mascara", mask)
    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

# -------------------------------------------------------------
# Análisis de movimiento
# -------------------------------------------------------------
centros = np.array(centros, dtype=float)
tiempos = np.array(tiempos)

# Escala promediada en cm/px
escala_cm_px = diametro_real_cm / np.mean(diametros_px)

# Sistema de referencia: Y = 0 en el suelo
alto_video_px = 960
centros_suelo = np.zeros_like(centros, dtype=float)
centros_suelo[:, 0] = centros[:, 0] * escala_cm_px
centros_suelo[:, 1] = (alto_video_px - centros[:, 1]) * escala_cm_px

# -------- Suavizar datos --------
def suavizar(datos, ventana=5):
    return np.convolve(datos, np.ones(ventana)/ventana, mode='valid')

x_suav = suavizar(centros_suelo[:, 0])
y_suav = suavizar(centros_suelo[:, 1])
centros_suav = np.vstack((x_suav, y_suav)).T
tiempos_suav = tiempos[:len(centros_suav)]

# Calcular velocidades y aceleraciones
velocidades = np.diff(centros_suav, axis=0) / dt
aceleraciones = np.diff(velocidades, axis=0) / dt
vel_magnitudes = np.linalg.norm(velocidades, axis=1)
acel_magnitudes = np.linalg.norm(aceleraciones, axis=1)

# -------------------------------------------------------------
# Velocidad inicial (vector y magnitud)
# -------------------------------------------------------------
v0_x, v0_y = velocidades[0]                 # componentes en cm/s (X horizontal, Y vertical hacia arriba)
v0_mag = np.linalg.norm([v0_x, v0_y])      # magnitud en cm/s

# Ángulo inicial de lanzamiento (coherente con suelo)
vx, vy = velocidades[0]
angulo_rad = np.arctan2(-vy, vx)
angulo_deg = np.degrees(angulo_rad)

# Altura inicial y posición final
altura_inicial_cm = centros_suelo[0, 1]
posicion_final_x = centros_suelo[-1, 0]
posicion_final_y = centros_suelo[-1, 1]

# -------------------------------------------------------------
# Resultados
# -------------------------------------------------------------
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
print(f"Velocidad inicial (centro): Vx = {v0_x:.2f} cm/s, Vy = {v0_y:.2f} cm/s, |V0| = {v0_mag:.2f} cm/s")
print(f"Velocidad promedio: {np.mean(vel_magnitudes):.2f} cm/s")
print(f"Aceleracion promedio: {np.mean(acel_magnitudes):.2f} cm/s²")
print(f"Ángulo inicial de lanzamiento: {angulo_deg:.2f}°")
print(f"Altura inicial (centro): {altura_inicial_cm:.2f} cm")
print(f"Posición final (centro): X = {posicion_final_x:.2f} cm, Y = {posicion_final_y:.2f} cm")

# -------------------------------------------------------------
# Gráfico de trayectoria
# -------------------------------------------------------------
plt.figure(figsize=(8, 6))
sc = plt.scatter(centros_suav[:, 0], centros_suav[:, 1], c=tiempos_suav,
                 cmap='viridis', s=40, label='Trayectoria (suavizada)')

# Línea de referencia en altura inicial
plt.axhline(y=altura_inicial_cm, color='red', linestyle='--',
            label=f'Altura inicial: {altura_inicial_cm:.1f} cm')

# Marcar la posición final
plt.scatter(posicion_final_x, posicion_final_y, color="blue", s=80, marker="x",
            label=f'Posición final ({posicion_final_x:.1f}, {posicion_final_y:.1f}) cm')

plt.colorbar(sc, label="Tiempo (s)")
plt.xlabel("X (cm)")
plt.ylabel("Altura Y (cm)")
plt.title("Trayectoria del centro de la pelota (referencia en el suelo)")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------------
# Velocidad vs tiempo
# -------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(tiempos_suav[1:], vel_magnitudes, marker="o", label="Velocidad")
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad (cm/s)")
plt.title("Velocidad de la pelota en función del tiempo")
plt.grid(True)
plt.legend()
plt.show()

# -------------------------------------------------------------
# Aceleración vs tiempo
# -------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(tiempos_suav[2:], acel_magnitudes, marker="o", color="red", label="Aceleración")
plt.xlabel("Tiempo (s)")
plt.ylabel("Aceleración (cm/s²)")
plt.title("Aceleración de la pelota en función del tiempo")
plt.grid(True)
plt.legend()
plt.show()

# -------------------------------------------------------------
# Comparación de trayectorias: píxeles vs centímetros
# -------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Trayectoria en píxeles (sistema original de OpenCV)
sc1 = axs[0].scatter(centros[:,0], centros[:,1], c=tiempos, cmap="plasma", s=30)
axs[0].invert_yaxis()  # porque OpenCV usa Y hacia abajo
axs[0].set_xlabel("X (px)")
axs[0].set_ylabel("Y (px)")
axs[0].set_title("Trayectoria en píxeles")
fig.colorbar(sc1, ax=axs[0], label="Tiempo (s)")

# Trayectoria en cm (sistema físico con suelo en Y=0)
sc2 = axs[1].scatter(centros_suav[:,0], centros_suav[:,1], c=tiempos_suav,
                     cmap="viridis", s=30)
axs[1].set_xlabel("X (cm)")
axs[1].set_ylabel("Altura Y (cm)")
axs[1].set_title("Trayectoria en cm (suavizada, suelo=0)")
fig.colorbar(sc2, ax=axs[1], label="Tiempo (s)")

plt.suptitle("Comparación de trayectorias: píxeles vs. centímetros", fontsize=14)
plt.show()