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

#desktop
alto_video_px = 960
ancho_video_px = 540

# laptop
#alto_video_px = 640
#ancho_video_px = 360

# Cargar el video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: no se pudo abrir el video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)     # Frames por segundo del video
dt = 1.0 / fps                      # Intervalo de tiempo entre frames

# Rango de color Lab para detectar la pelota verde
lower_green = np.array([60, 70, 154])
upper_green = np.array([255, 130, 255])

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

        frame = cv2.resize(frame, (ancho_video_px, alto_video_px))

        # Convertir a espacio Lab y aplicar máscara por rango de color
        Lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        mask = cv2.inRange(Lab, lower_green, upper_green)

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

                diametros_px.append(2 * radio)  # guardar todos

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
diametros_px = np.array(diametros_px, dtype=float)

# Escala global en cm/px
escala_cm_px = diametro_real_cm / np.mean(diametros_px)


centros_suelo = np.zeros_like(centros, dtype=float)
centros_suelo[:, 0] = centros[:, 0] * escala_cm_px
centros_suelo[:, 1] = (alto_video_px - centros[:, 1]) * escala_cm_px

#Ajuste: poner el cero en la última posición de la pelota
y_final = centros_suelo[-1, 1]     # altura de la última posición detectada
centros_suelo[:, 1] = centros_suelo[:, 1] - y_final

x_suav = centros_suelo[:, 0]
y_suav = centros_suelo[:, 1]

# -------------------------------------------------------------
# Cálculo de velocidades y aceleraciones
# -------------------------------------------------------------
vx = np.gradient(x_suav, tiempos)
vy = np.gradient(y_suav, tiempos)
vel_magnitudes = np.sqrt(vx**2 + vy**2)

ax = np.gradient(vx, tiempos)
ay = np.gradient(vy, tiempos)
acel_magnitudes = np.sqrt(ax**2 + ay**2)

# -------------------------------------------------------------
# Estimación robusta de condiciones iniciales
# -------------------------------------------------------------
g = 981.0
coef_x = np.polyfit(tiempos, x_suav, 1)
v0_x = coef_x[0]
x_inicial = coef_x[1]

y_adj = y_suav + 0.5 * g * tiempos**2
coef_y = np.polyfit(tiempos, y_adj, 1)
v0_y = coef_y[0]
altura_inicial = coef_y[1]

v0 = np.linalg.norm([v0_x, v0_y])
angulo_inicial = np.degrees(np.arctan2(v0_y, v0_x))

disc = v0_y**2 + 2 * g * altura_inicial
tiempo_vuelo = (v0_y + np.sqrt(disc)) / g if disc >= 0 else tiempos[-1]

alcance_maximo = v0_x * tiempo_vuelo
altura_maxima = altura_inicial + (v0_y**2) / (2 * g)

# -------------------------------------------------------------
# Trayectoria teórica en cm (limpia y ajustada)
# -------------------------------------------------------------
t_teorico = np.linspace(0, tiempo_vuelo, 300)
x_teorico = x_inicial + v0_x * t_teorico
y_teorico = altura_inicial + v0_y * t_teorico - 0.5 * g * t_teorico**2


# -------------------------------------------------------------
# Convertir trayectoria teórica a píxeles para el video
# -------------------------------------------------------------
# Coordenadas finales reales (en píxeles)
final_real_px = (int(centros[-1][0]), int(centros[-1][1]))

# Coordenadas finales teóricas (en píxeles, sin ajustar)
final_teo_px = (
    int(x_teorico[-1] / escala_cm_px),
    int(alto_video_px - (y_teorico[-1] / escala_cm_px))
)

# Desplazamiento en píxeles necesario para que coincidan con el suelo
dy = final_real_px[1] - final_teo_px[1]

# Aplicar desplazamiento a toda la curva teórica
x_teo_px = (x_teorico / escala_cm_px).astype(int)
y_teo_px = (alto_video_px - (y_teorico / escala_cm_px)).astype(int) + dy

# Limitar dentro del frame
x_teo_px = np.clip(x_teo_px, 0, ancho_video_px - 1)
y_teo_px = np.clip(y_teo_px, 0, alto_video_px - 1)

# -------------------------------------------------------------
# Resultados numéricos
# -------------------------------------------------------------
print("\n--- RESULTADOS DEL ANÁLISIS DE MOVIMIENTO ---")
print(f"Escala: 1 px = {escala_cm_px:.3f} cm")
print(f"Total de frames analizados: {len(centros)}")

print("\nVelocidades por frame (cm/s):")
for i, v in enumerate(vel_magnitudes):
    print(f"t = {tiempos[i]:.3f} s -> {v:.2f} cm/s")

print("\nAceleraciones por frame (cm/s²):")
for i, a in enumerate(acel_magnitudes):
    print(f"t = {tiempos[i]:.3f} s -> {a:.2f} cm/s²")

print("\nResumen global:")
print(f"Velocidad inicial (centro): |V0| = {v0:.2f} cm/s")
print(f"Velocidad promedio: {np.mean(vel_magnitudes):.2f} cm/s")
print(f"Aceleración promedio: {np.mean(acel_magnitudes):.2f} cm/s²")
print(f"Ángulo inicial de lanzamiento: {angulo_inicial:.2f}°")
print(f"Altura inicial (centro): {altura_inicial:.2f} cm")

# Posición final medida
posicion_final_x = x_suav[-1]
posicion_final_y = y_suav[-1]
print(f"Posición final (centro): X = {posicion_final_x:.2f} cm, Y = {posicion_final_y:.2f} cm")

# Extra: datos teóricos
print("\nResultados teóricos:")
print(f"Tiempo de vuelo: {tiempo_vuelo:.3f} s")
print(f"Alcance máximo: {alcance_maximo:.2f} cm")
print(f"Altura máxima: {altura_maxima:.2f} cm")

# -------------------------------------------------------------
# Crear video con resultados
# -------------------------------------------------------------
print("\n--- CREANDO VIDEO CON ANÁLISIS ---")
cap = cv2.VideoCapture(video_path)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video_analizado.mp4', fourcc, fps, (ancho_video_px, alto_video_px))

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (ancho_video_px, alto_video_px))

    if frame_idx < len(tiempos):
        t = tiempos[frame_idx]
        vel_x = vx[frame_idx] if frame_idx < len(vx) else 0
        vel_y = vy[frame_idx] if frame_idx < len(vy) else 0
        vel_total = vel_magnitudes[frame_idx] if frame_idx < len(vel_magnitudes) else 0

        info_lines = [
            f"Tiempo: {t:.2f}s",
            f"Vel X: {vel_x:.1f}cm/s",
            f"Vel Y: {vel_y:.1f}cm/s",
            f"Vel Total: {vel_total:.1f}cm/s",
            "",
            "PARAMETROS INICIALES",
            f"Vel inicial: {v0:.1f}cm/s",
            f"Angulo: {angulo_inicial:.1f} grados",
            f"H inicial: {altura_inicial:.1f}cm",
            "",
            "TEORICOS:",
            f"T vuelo: {tiempo_vuelo:.2f}s",
            f"Alcance: {alcance_maximo:.1f}cm",
            f"H max: {altura_maxima:.1f}cm"
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (340, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        if frame_idx < len(x_suav):
            centro_px = (int(centros[frame_idx][0]), int(centros[frame_idx][1]))
            cv2.circle(frame, centro_px, 3, (0, 0, 255), -1)  # punto real

            escala_vel = 1
            end_x = int(centros[frame_idx][0] + vel_x * escala_vel)
            end_y = int(centros[frame_idx][1] - vel_y * escala_vel)
            cv2.arrowedLine(frame, centro_px, (end_x, end_y), (0, 255, 0), 3)

        # Dibujar curva teórica
        for i in range(len(x_teo_px) - 1):
            cv2.line(frame,
                     (x_teo_px[i], y_teo_px[i]),
                     (x_teo_px[i+1], y_teo_px[i+1]),
                     (0, 255, 255), 3)

        # Marcar el punto final real (verde) y teórico (rojo)
        final_real_px = (int(centros[-1][0]), int(centros[-1][1]))
        final_teo_px = (x_teo_px[-1], y_teo_px[-1])
        cv2.circle(frame, final_real_px, 6, (0, 255, 0), -1)   # verde sólido
        cv2.drawMarker(frame, final_teo_px, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 15, 2)

    frame_idx += 1
    out.write(frame)

cap.release()
out.release()
print("Video analizado guardado como 'video_analizado.mp4'")

# -------------------------------------------------------------
# Gráficas de análisis con matplotlib
# -------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(x_suav, y_suav, "o-", label="Trayectoria obtenida")
plt.plot(x_teorico, y_teorico, "r--", label="Trayectoria teórica")
plt.axhline(y=0, color="k", linestyle="--")

# Marcar el punto final (obtenida y teórico) para ver que coinciden
plt.plot(x_suav[-1], y_suav[-1], "go", markersize=10, label="Final obtenido")
plt.plot(x_teorico[-1], y_teorico[-1], "rx", markersize=10, label="Final teórico")

plt.xlabel("X (cm)")
plt.ylabel("Altura Y (cm)")
plt.title("Comparación de trayectorias")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(tiempos, vel_magnitudes, "b-o", label="|V| magnitud")
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad (cm/s)")
plt.title("Velocidad en función del tiempo")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(x_suav, y_suav, "o-", label="Trayectoria obtenida")
plt.quiver(x_suav, y_suav, ax, ay, color="red", angles="xy",
           scale_units="xy", scale=300,width=0.004, label="Vectores de aceleración")
plt.xlabel("X (cm)")
plt.ylabel("Altura Y (cm)")
plt.title("Trayectoria con vectores de aceleración")
plt.legend()
plt.grid(True)
plt.show()