import cv2
import numpy as np
from matplotlib import pyplot as plt

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

# Rango de color HSV para detectar la pelota verde
lower_green = np.array([5, 50, 60])
upper_green = np.array([168, 255, 255])

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
        # frame = cv2.resize(frame, (540, 960))   #desktop
        frame = cv2.resize(frame, (360, 640))  # laptop

        # Convertir a espacio HSV y aplicar máscara por rango de color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Procesamiento morfológico para eliminar ruido en la máscara
        kernel_small = np.ones((3, 3), np.uint8)    # Kernel pequeño para eliminar puntos de ruido
        kernel_large = np.ones((7, 7), np.uint8)    # Kernel grande para cerrar huecos


        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small) # "Opening" elimina pequeños objetos irrelevantes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)# "Closing" rellena huecos dentro de la región detectada

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
alto_video_px = 640
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
vx = velocidades[:, 0]
vy = velocidades[:, 1]
vel_magnitudes = np.linalg.norm(velocidades, axis=1)

aceleraciones = np.diff(velocidades, axis=0) / dt
acel_magnitudes = np.linalg.norm(aceleraciones, axis=1)

# Condiciones iniciales
v0_x, v0_y = velocidades[0]
v0 = np.linalg.norm([v0_x, v0_y])
angulo_inicial = np.degrees(np.arctan2(v0_y, v0_x))
altura_inicial = centros_suelo[0, 1]

# Física teórica
g = 981
tiempo_vuelo = (v0_y + np.sqrt(v0_y**2 + 2 * g * altura_inicial)) / g
alcance_maximo = v0_x * tiempo_vuelo
altura_maxima = altura_inicial + (v0_y**2) / (2 * g)

# Generar curva teórica
t_teorico = np.linspace(0, tiempo_vuelo, 200)
x_teorico = v0_x * t_teorico
y_teorico = altura_inicial + v0_y * t_teorico - 0.5 * g * t_teorico**2
trayectoria_teorica = np.vstack((x_teorico, y_teorico)).T


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
print(f"Velocidad inicial (centro): |V0| = {v0:.2f} cm/s")
print(f"Velocidad promedio: {np.mean(vel_magnitudes):.2f} cm/s")
print(f"Aceleracion promedio: {np.mean(acel_magnitudes):.2f} cm/s²")
print(f"Ángulo inicial de lanzamiento: {angulo_inicial:.2f}°")
print(f"Altura inicial (centro): {altura_inicial:.2f} cm")

# -------------------------------------------------------------
# Gráficos de resultados
# -------------------------------------------------------------

# Trayectoria suavizada
plt.figure(figsize=(8, 6))
sc = plt.scatter(centros_suav[:, 0], centros_suav[:, 1], c=tiempos_suav,
                 cmap='viridis', s=40, label='Trayectoria (suavizada)')
plt.axhline(y=altura_inicial, color='red', linestyle='--',                   # Línea de referencia en altura inicial
            label=f'Altura inicial: {altura_inicial:.1f} cm')
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

# -------------------------------------------------------------
# Crear video con información y trayectoria teórica
# -------------------------------------------------------------
print("\n--- CREANDO VIDEO CON ANÁLISIS ---")

cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video_analizado.mp4', fourcc, fps, (720, 640))

# Función para pasar de cm (mundo físico) a px (video)
def cm_a_px(x_cm, y_cm, escala_cm_px, alto_video_px=640):
    x_px = int(x_cm / escala_cm_px)
    y_px = int(alto_video_px - (y_cm / escala_cm_px))
    return x_px, y_px

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (360, 640))
    info_panel = np.zeros((640, 360, 3), dtype=np.uint8)
    combined_frame = np.hstack([frame, info_panel])

    if frame_idx < len(tiempos):
        t = tiempos[frame_idx]
        if frame_idx < len(velocidades):
            vel_x, vel_y = velocidades[frame_idx]
            vel_total = vel_magnitudes[frame_idx]
        else:
            vel_x, vel_y, vel_total = 0, 0, 0

        # Info panel
        info_lines = [
            f"Tiempo: {t:.2f} s",
            f"Vel X: {vel_x:.1f} cm/s",
            f"Vel Y: {vel_y:.1f} cm/s",
            f"Vel Total: {vel_total:.1f} cm/s",
            "",
            "PARAMETROS INICIALES:",
            f"V0: {v0:.1f} cm/s",
            f"Angulo: {angulo_inicial:.1f}°",
            f"H0: {altura_inicial:.1f} cm",
            "",
            "TEORICOS:",
            f"T vuelo: {tiempo_vuelo:.2f} s",
            f"Alcance: {alcance_maximo:.1f} cm",
            f"H max: {altura_maxima:.1f} cm"
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(combined_frame, line, (370, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Pelota real y vector velocidad
        if frame_idx < len(centros):
            centro_px = (int(centros[frame_idx][0]), int(centros[frame_idx][1]))
            cv2.circle(combined_frame, centro_px, 3, (0, 0, 255), -1)

            escala_vel = 2
            end_x = int(centros[frame_idx][0] + vel_x * escala_vel)
            end_y = int(centros[frame_idx][1] - vel_y * escala_vel)
            cv2.arrowedLine(combined_frame, centro_px,
                            (end_x, end_y), (0, 255, 0), 2)

        # Trayectoria teórica
        for i in range(len(trayectoria_teorica) - 1):
            x1, y1 = cm_a_px(trayectoria_teorica[i, 0], trayectoria_teorica[i, 1], escala_cm_px, 640)
            x2, y2 = cm_a_px(trayectoria_teorica[i+1, 0], trayectoria_teorica[i+1, 1], escala_cm_px, 640)
            cv2.line(combined_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    frame_idx += 1
    out.write(combined_frame)

    if frame_idx % 30 == 0:
        print(f"Procesando frame {frame_idx}/{len(tiempos)}")

cap.release()
out.release()

print("✅ Video analizado guardado como 'video_analizado.mp4'")