import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# Control de flujo del video
# -------------------------------------------------------------
pausar_video = False
frame_a_frame = False

# Función callback para controlar la reproducción con el mouse
def mouse_callback(event, x, y, flags, param):
    global pausar_video, frame_a_frame
    if event == cv2.EVENT_RBUTTONDOWN:  # Pausar/reanudar con clic derecho
        pausar_video = not pausar_video
    if event == cv2.EVENT_LBUTTONDOWN:  # Avanzar un frame cuando está pausado
        if pausar_video:
            frame_a_frame = True

# -------------------------------------------------------------
# Parámetros de entrada
# -------------------------------------------------------------
video_path = "../Videos Procesados/V1_procesado.mp4"
diametro_real_cm = 9

# Cargar el video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: no se pudo abrir el video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)     # Frames por segundo del video
dt = 1.0 / fps                      # Intervalo de tiempo entre frames

# Rango de color LAB para detectar la pelota verde
lower_lab = np.array([42, 90, 150])
upper_lab = np.array([121, 118, 255])

# Listas para guardar datos de la trayectoria
tiempos = []
centros = []
diametros_px = []   # Para calcular escala más estable
frame_idx = 0

# Configurar ventana interactiva
cv2.namedWindow("Video")
cv2.setMouseCallback("Video", mouse_callback)

# -------------------------------------------------------------
# Procesamiento del video
# -------------------------------------------------------------
while True:
    if not pausar_video or frame_a_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionamos video a tamaño fijo
        frame = cv2.resize(frame, (540, 960))

        # Convertir a espacio HSV y aplicar máscara por rango de color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        mask = cv2.inRange(hsv, lower_lab, upper_lab)

        # Procesamiento morfológico para eliminar ruido en la máscara
        kernel_small = np.ones((3, 3), np.uint8)    # Kernel pequeño para eliminar puntos de ruido
        kernel_large = np.ones((7, 7), np.uint8)    # Kernel grande para cerrar huecos


        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small) # "Opening" elimina pequeños objetos irrelevantes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)# "Closing" rellena huecos dentro de la región detectada
        mask = cv2.medianBlur(mask, 5)                         # Filtro de mediana para suavizar bordes de la máscara

        # Encontrar contornos y calcular centro de la pelota
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area > 500:  # Filtrar objetos pequeños irrelevantes
                (cx, cy), radio = cv2.minEnclosingCircle(contorno)
                centro = (int(cx), int(cy))
                radio = int(radio)

                # Guardar diámetro en píxeles de los primeros frames para estimar escala
                if frame_idx < 5:
                    diametros_px.append(2 * radio)

                # Guardar posición y tiempo
                tiempos.append(frame_idx * dt)
                centros.append((centro[0], centro[1]))

                # Dibujar marcador en la pelota
                cv2.circle(frame, centro, radio, (0, 255, 0), 2)
                cv2.circle(frame, centro, 3, (0, 0, 255), -1)

        frame_idx += 1
        frame_a_frame = False

    # Mostrar video y máscara en tiempo real
    cv2.imshow("Video", frame)
    cv2.imshow("Mascara", mask)
    if cv2.waitKey(10) & 0xFF == 27:    # Tecla ESC para salir
        break

cap.release()
cv2.destroyAllWindows()

# -------------------------------------------------------------
# Análisis de movimiento
# -------------------------------------------------------------
centros = np.array(centros, dtype=float)
tiempos = np.array(tiempos)

# Calcular escala en cm/px usando el diámetro real
escala_cm_px = diametro_real_cm / np.mean(diametros_px)

# Convertir coordenadas a sistema físico (Y=0 en el suelo)
alto_video_px = 960
centros_suelo = np.zeros_like(centros, dtype=float)
centros_suelo[:, 0] = centros[:, 0] * escala_cm_px
centros_suelo[:, 1] = (alto_video_px - centros[:, 1]) * escala_cm_px

# Función para suavizar señales (filtro de ventana móvil)
def suavizar(datos, ventana=5):
    return np.convolve(datos, np.ones(ventana)/ventana, mode='valid')

# Suavizado de posiciones
x_suav = suavizar(centros_suelo[:, 0])
y_suav = suavizar(centros_suelo[:, 1])
centros_suav = np.vstack((x_suav, y_suav)).T
tiempos_suav = tiempos[:len(centros_suav)]

# Calcular velocidades, aceleraciones y magnitudes
velocidades = np.diff(centros_suav, axis=0) / dt
aceleraciones = np.diff(velocidades, axis=0) / dt
vel_magnitudes = np.linalg.norm(velocidades, axis=1)
acel_magnitudes = np.linalg.norm(aceleraciones, axis=1)

# Velocidad inicial (vector y magnitud)
v0_x, v0_y = velocidades[0]
v0_mag = np.linalg.norm([v0_x, v0_y])

# Ángulo inicial de lanzamiento (positivo hacia arriba)
vx, vy = velocidades[0]
angulo_rad = np.arctan2(-vy, vx)
angulo_deg = np.degrees(angulo_rad)

# Altura inicial y posición final
altura_inicial_cm = centros_suelo[0, 1]
posicion_final_x = centros_suelo[-1, 0]
posicion_final_y = centros_suelo[-1, 1]

# -------------------------------------------------------------
# Resultados numéricos
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
print(f"Velocidad inicial (centro): |V0| = {v0_mag:.2f} cm/s")
print(f"Velocidad promedio: {np.mean(vel_magnitudes):.2f} cm/s")
print(f"Aceleracion promedio: {np.mean(acel_magnitudes):.2f} cm/s²")
print(f"Ángulo inicial de lanzamiento: {angulo_deg:.2f}°")
print(f"Altura inicial (centro): {altura_inicial_cm:.2f} cm")
print(f"Posición final (centro): X = {posicion_final_x:.2f} cm, Y = {posicion_final_y:.2f} cm")

# -------------------------------------------------------------
# Gráficos de resultados
# -------------------------------------------------------------

# Trayectoria suavizada
plt.figure(figsize=(8, 6))
sc = plt.scatter(centros_suav[:, 0], centros_suav[:, 1], c=tiempos_suav,
                 cmap='viridis', s=40, label='Trayectoria (suavizada)')
plt.axhline(y=altura_inicial_cm, color='red', linestyle='--',                   # Línea de referencia en altura inicial
            label=f'Altura inicial: {altura_inicial_cm:.1f} cm')
plt.scatter(posicion_final_x, posicion_final_y, color="blue", s=80, marker="x", # Marcar la posición final
            label=f'Posición final ({posicion_final_x:.1f}, {posicion_final_y:.1f}) cm')
plt.colorbar(sc, label="Tiempo (s)")
plt.xlabel("X (cm)")
plt.ylabel("Altura Y (cm)")
plt.title("Trayectoria del centro de la pelota (referencia en el suelo)")
plt.legend()
plt.grid(True)
plt.show()

# Velocidad en función del tiempo
plt.figure(figsize=(8, 4))
plt.plot(tiempos_suav[1:], vel_magnitudes, marker="o", label="Velocidad")
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad (cm/s)")
plt.title("Velocidad de la pelota en función del tiempo")
plt.grid(True)
plt.legend()
plt.show()

# Aceleración en función del tiempo
plt.figure(figsize=(8, 4))
plt.plot(tiempos_suav[2:], acel_magnitudes, marker="o", color="red", label="Aceleración")
plt.xlabel("Tiempo (s)")
plt.ylabel("Aceleración (cm/s²)")
plt.title("Aceleración de la pelota en función del tiempo")
plt.grid(True)
plt.legend()
plt.show()

# Comparación de trayectorias en píxeles vs. centímetros
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