import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# =============================================================
# CONFIGURACIÓN INICIAL
# =============================================================

# Variables para controlar la reproducción del video
pausar_video = False
frame_a_frame = False

# Función callback para controlar la reproducción con el mouse
def mouse_callback(event, x, y, flags, param):
    global pausar_video, frame_a_frame
    if event == cv2.EVENT_RBUTTONDOWN:  # Click derecho -> Pausar/reanudar
        pausar_video = not pausar_video
    if event == cv2.EVENT_LBUTTONDOWN:  # Click izquierdo -> Avanza un frame
        if pausar_video:
            frame_a_frame = True

# Ruta del video y parámetros de referencia
video_path = "../Videos Procesados/V1_procesado.mp4"
diametro_real_cm = 9

# Resolución de salida del video (ajustar según PC)
alto_video_px = 960
ancho_video_px = 540

# Cargar el video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: no se pudo abrir el video")
    exit()

# Calcular FPS y delta t (tiempo entre frames)
fps = cap.get(cv2.CAP_PROP_FPS)
dt = 1.0 / fps

# Rango de color en espacio Lab para detectar la pelota verde
lower_green = np.array([60, 70, 154])
upper_green = np.array([255, 150, 255])

# Listas donde guardaremos los resultados del análisis
tiempos = []        # tiempo de cada frame
centros = []        # coordenadas (x,y) de la pelota
diametros_px = []   # diámetros detectados para calcular escala
frame_idx = 0

# Configurar ventana de OpenCV y callback de mouse
cv2.namedWindow("Video")
cv2.setMouseCallback("Video", mouse_callback)

# =============================================================
# DETECCIÓN DE LA PELOTA EN EL VIDEO
# =============================================================
while True:
    if not pausar_video or frame_a_frame:
        ret, frame = cap.read()
        if not ret: # Se terminó el video
            break
        # Redimensionamos cada frame al tamaño definido
        frame = cv2.resize(frame, (ancho_video_px, alto_video_px))

        # Convertir a espacio Lab y aplicar máscara por rango de color
        Lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
        mask = cv2.inRange(Lab, lower_green, upper_green)

        # Procesamiento morfológico para eliminar ruido en la máscara
        kernel_small = np.ones((3, 3), np.uint8)    # Kernel pequeño para eliminar puntos de ruido
        kernel_large = np.ones((7, 7), np.uint8)    # Kernel grande para cerrar huecos


        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small) # "Opening" elimina pequeños objetos irrelevantes
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)# "Closing" rellena huecos dentro de la región detectada
        mask = cv2.medianBlur(mask, 5)                         # Filtro para suavizar bordes de la máscara

        # Encontrar contornos y calcular centro de la pelota
        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area > 500:  # Filtrar objetos pequeños irrelevantes
                (cx, cy), radio = cv2.minEnclosingCircle(contorno)
                centro = (int(cx), int(cy))
                radio = int(radio)

                diametros_px.append(2 * radio)          # guardar diámetro para calcular escala
                tiempos.append(frame_idx * dt)          # guardar instante de tiempo
                centros.append((centro[0], centro[1]))  # guardar posición

                # Dibujar marcador en la pelota
                cv2.circle(frame, centro, radio, (0, 255, 0), 4)
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

# =============================================================
# ANÁLISIS DE MOVIMIENTO
# =============================================================

# Convertimos listas a arreglos NumPy
centros = np.array(centros, dtype=float)
tiempos = np.array(tiempos)
diametros_px = np.array(diametros_px, dtype=float)

# Escala global: cuántos cm equivale cada píxel
escala_cm_px = diametro_real_cm / np.mean(diametros_px)

# Convertimos coordenadas a sistema físico (en cm, con Y=0 en el suelo)
centros_suelo = np.zeros_like(centros, dtype=float)
centros_suelo[:, 0] = centros[:, 0] * escala_cm_px
centros_suelo[:, 1] = (alto_video_px - centros[:, 1]) * escala_cm_px

# Ajuste: que el último punto detectado sea Y=0 (suelo)
y_final = centros_suelo[-1, 1]      # altura de la última posición detectada
centros_suelo[:, 1] -= y_final

# Suavizado de trayectorias usando filtro de Savitzky-Golay
x_suav = savgol_filter(centros_suelo[:, 0], window_length=15, polyorder=2)
y_suav = savgol_filter(centros_suelo[:, 1], window_length=15, polyorder=2)

# =============================================================
# CÁLCULO DE VELOCIDADES Y ACELERACIONES
# =============================================================
vx = np.gradient(x_suav, tiempos)
vy = np.gradient(y_suav, tiempos)
vel_magnitudes = np.sqrt(vx**2 + vy**2)

ax = np.gradient(vx, tiempos)
ay = np.gradient(vy, tiempos)
acel_magnitudes = np.sqrt(ax**2 + ay**2)

# =============================================================
# ESTIMACIÓN DE PARÁMETROS INICIALES
# =============================================================
g = 981.0  # gravedad (cm/s²)


# Componente horizontal (X)
# Se ajusta una recta X(t) ≈ v0_x * t + x_inicial
coef_x = np.polyfit(tiempos, x_suav, 1)
v0_x = coef_x[0]        # Velocidad inicial en X (cm/s)
x_inicial = coef_x[1]   # Posición inicial en X (cm)

# Componente vertical (Y)
# En Y hay aceleración (gravedad). Para linealizar se suma el término (1/2 * g * t²):
# Y(t) + (1/2 * g * t²) ≈ v0_y * t + y_inicial
y_adj = y_suav + 0.5 * g * tiempos**2
coef_y = np.polyfit(tiempos, y_adj, 1)
v0_y = coef_y[0]           # Velocidad inicial en Y (cm/s)
altura_inicial = coef_y[1] # Altura inicial (cm)

# Magnitud de la velocidad inicial y ángulo
v0 = np.linalg.norm([v0_x, v0_y])                    # Velocidad inicial total (vectorial)
angulo_inicial = np.degrees(np.arctan2(v0_y, v0_x))  # Ángulo en grados respecto al eje X

# Tiempo total de vuelo, alcance máximo y altura máxima
disc = v0_y**2 + 2 * g * altura_inicial
tiempo_vuelo = (v0_y + np.sqrt(disc)) / g if disc >= 0 else tiempos[-1]
alcance_maximo = v0_x * tiempo_vuelo
altura_maxima = altura_inicial + (v0_y**2) / (2 * g)

# =============================================================
# TRAYECTORIA TEÓRICA
# =============================================================
# Generar un conjunto de tiempos para suavizar la curva teórica
t_teorico = np.linspace(0, tiempo_vuelo, 300)

# Ecuaciones de la parábola
x_teorico = x_inicial + v0_x * t_teorico
y_teorico = altura_inicial + v0_y * t_teorico - 0.5 * g * t_teorico**2

# Velocidades teóricas
vx_teorico = np.gradient(x_teorico, t_teorico)
vy_teorico = np.gradient(y_teorico, t_teorico)
vel_teorico = np.sqrt(vx_teorico**2 + vy_teorico**2)

# Aceleraciones teóricas
ax_teorico = np.gradient(vx_teorico, t_teorico)
ay_teorico = np.gradient(vy_teorico, t_teorico)
acel_teorico = np.sqrt(ax_teorico**2 + ay_teorico**2)

# =============================================================
# CONVERTIR TRAYECTORIA A PÍXELES PARA EL VIDEO
# =============================================================

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

# =============================================================
# MOSTRAR RESULTADOS EN CONSOLA
# =============================================================
print("\n--- RESULTADOS DEL ANÁLISIS DE MOVIMIENTO ---")
print(f"Escala: 1 px = {escala_cm_px:.3f} cm")
print(f"Total de frames analizados: {len(centros)}")
print(f"Velocidad inicial: |V0| = {v0:.2f} cm/s")
print(f"Ángulo inicial: {angulo_inicial:.2f}°")
print(f"Altura inicial: {altura_inicial:.2f} cm")

# -------------------------------
# Valores obtenidos del experimento
# -------------------------------
tiempo_vuelo_obt = tiempos[-1] - tiempos[0]
vel_prom_obt = np.mean(vel_magnitudes)
acel_prom_obt = np.mean(acel_magnitudes)
altura_max_obt = np.max(y_suav)
alcance_max_obt = x_suav[-1]

print("\nValores obtenidos:")
print(f"Tiempo de vuelo: {tiempo_vuelo_obt:.3f} s")
print(f"Velocidad promedio: {vel_prom_obt:.2f} cm/s")
print(f"Aceleración promedio: {acel_prom_obt:.2f} cm/s²")
print(f"Altura máxima: {altura_max_obt:.2f} cm")
print(f"Alcance máximo: {alcance_max_obt:.2f} cm")

# -------------------------------
# Valores esperados teóricos
# -------------------------------
vel_prom_teo = np.mean(vel_teorico)
acel_prom_teo = np.mean(acel_teorico)

print("\nEsperados teóricos:")
print(f"Tiempo de vuelo: {tiempo_vuelo:.3f} s")
print(f"Velocidad promedio: {vel_prom_teo:.2f} cm/s")
print(f"Aceleración promedio: {acel_prom_teo:.2f} cm/s²")
print(f"Altura máxima: {altura_maxima:.2f} cm")
print(f"Alcance máximo: {alcance_maximo:.2f} cm")

# -------------------------------
# Errores porcentuales
# -------------------------------
def error_porcentual(obt, teo):
    return abs((obt - teo) / teo) * 100 if teo != 0 else float('nan')

print("\nErrores porcentuales:")
print(f"Tiempo de vuelo: {error_porcentual(tiempo_vuelo_obt, tiempo_vuelo):.2f}%")
print(f"Velocidad promedio: {error_porcentual(vel_prom_obt, vel_prom_teo):.2f}%")
print(f"Aceleración promedio: {error_porcentual(acel_prom_obt, acel_prom_teo):.2f}%")
print(f"Altura máxima: {error_porcentual(altura_max_obt, altura_maxima):.2f}%")
print(f"Alcance máximo: {error_porcentual(alcance_max_obt, alcance_maximo):.2f}%")


# =============================================================
# CREAR VIDEO CON RESULTADOS
# =============================================================
print("\n--- CREANDO VIDEO CON ANÁLISIS ---")

# Reabrimos el video original
cap = cv2.VideoCapture(video_path)

# Definimos el códec y el objeto para guardar el nuevo video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video_analizado.mp4', fourcc, fps, (ancho_video_px, alto_video_px))

frame_idx = 0
suma_acel = 0
while True:
    ret, frame = cap.read()
    if not ret: # Se terminó el video
        break
    # Redimensionamos cada frame al tamaño definido
    frame = cv2.resize(frame, (ancho_video_px, alto_video_px))

    # Solo dibujamos datos si existen cálculos para este frame
    if frame_idx < len(tiempos):
        # Recuperar valores físicos correspondientes a este frame
        t = tiempos[frame_idx]      # tiempo actual
        vel_x = vx[frame_idx] if frame_idx < len(vx) else 0
        vel_y = vy[frame_idx] if frame_idx < len(vy) else 0
        vel_total = vel_magnitudes[frame_idx] if frame_idx < len(vel_magnitudes) else 0

        # Aceleración instantánea en este frame
        acel_instant = acel_magnitudes[frame_idx] if frame_idx < len(acel_magnitudes) else 0

        # Actualizar promedio hasta este frame
        suma_acel += acel_instant
        acel_promedio = suma_acel / (frame_idx + 1)

        # Posición suavizada en centímetros
        pos_x_cm = x_suav[frame_idx] if frame_idx < len(x_suav) else x_suav[-1]
        pos_y_cm = y_suav[frame_idx] if frame_idx < len(y_suav) else y_suav[-1]

        # Resultados teóricos de referencia
        pos_final_teo_x = alcance_maximo
        pos_final_teo_y = 0.0
        acel_teorica = g

        # Texto que se mostrará en pantalla
        info_lines = [
            # Datos instantáneos
            "DATOS INSTANTANEOS:",
            f"Tiempo actual: {t:.2f} s",
            f"Velocidad: {vel_total:.1f} cm/s",
            f"Aceleracion: {acel_instant:.1f} cm/s2",
            f"Acel. prom.: {acel_promedio:.1f} cm/s2",
            f"Posicion(X,Y): ({pos_x_cm:.1f}, {pos_y_cm:.1f}) cm",
            "",
            # Parámetros iniciales del lanzamiento
            "PARAMETROS INICIALES:",
            f"Vel inicial: {v0:.1f} cm/s",
            f"Altura inicial: {altura_inicial:.1f} cm",
            f"Angulo inicial: {angulo_inicial:.1f} deg",
            "",
            # Resultados teóricos esperados
            "ESPERADO TEORICO:",
            f"Tiempo de vuelo: {tiempo_vuelo:.2f} s",
            f"Aceleracion: {acel_teorica:.1f} cm/s2",
            f"Altura maxima: {altura_maxima:.1f} cm",
            f"Posicion final: X={pos_final_teo_x:.1f} cm"
        ]
        # Dibujar el texto en el frame (columna derecha)
        x_text = 270
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (x_text, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # Dibujar punto real y vector de velocidad
        if frame_idx < len(centros):
            centro_px = (int(centros[frame_idx][0]), int(centros[frame_idx][1]))
            cv2.circle(frame, centro_px, 3, (0, 0, 255), -1)  # punto real

            # Flecha que representa la velocidad instantánea
            escala_vel = 1  # si quieres escalar la flecha, cambiar aquí
            end_x = int(centros[frame_idx][0] + vel_x * escala_vel)
            end_y = int(centros[frame_idx][1] - vel_y * escala_vel)
            cv2.arrowedLine(frame, centro_px, (end_x, end_y), (0, 255, 0), 3)

        # Dibujar trayectoria teórica (curva amarilla)
        for i in range(len(x_teo_px) - 1):
            cv2.line(frame,
                     (x_teo_px[i], y_teo_px[i]),
                     (x_teo_px[i+1], y_teo_px[i+1]),
                     (0, 255, 255), 3)

        # Marcar punto final real (verde) y teórico (rojo)
        final_real_px = (int(centros[-1][0]), int(centros[-1][1]))
        final_teo_px = (x_teo_px[-1], y_teo_px[-1])
        cv2.circle(frame, final_real_px, 6, (0, 255, 0), -1)
        cv2.drawMarker(frame, final_teo_px, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 15, 2)

    # Pasamos al siguiente frame y lo guardamos en el nuevo video
    frame_idx += 1
    out.write(frame)

# Liberar recursos
cap.release()
out.release()
print("Video analizado guardado como 'video_analizado.mp4'")

# =============================================================
# GRÁFICAS DE ANÁLISIS CON MATPLOTLIB
# =============================================================
# 1) Comparación de trayectorias (real vs teórica)
plt.figure(figsize=(8, 6))
plt.plot(x_suav, y_suav, "o-", label="Trayectoria obtenida")        # Trayectoria real obtenida del video
plt.plot(x_teorico, y_teorico, "r--", label="Trayectoria teórica") # Trayectoria calculada teóricamente con física
plt.axhline(y=0, color="k", linestyle="--")                             # Línea horizontal en Y=0 (suelo de referencia)

# Marcar punto final real (verde) y punto final teórico (rojo)
plt.plot(x_suav[-1], y_suav[-1], "go", markersize=10, label="Final obtenido")
plt.plot(x_teorico[-1], y_teorico[-1], "rx", markersize=10, label="Final teórico")

plt.xlabel("X (cm)")
plt.ylabel("Altura Y (cm)")
plt.title("Comparación de trayectorias")
plt.legend()
plt.grid(True)
plt.show()

# 2) Velocidad en función del tiempo
plt.figure(figsize=(8, 4))
plt.plot(tiempos, vel_magnitudes, "bo-", label="Velocidad obtenida")
plt.plot(t_teorico, vel_teorico, "r--", label="Velocidad teórica")
plt.xlabel("Tiempo (s)")
plt.ylabel("Velocidad (cm/s)")
plt.title("Comparación: Velocidad obtenida vs teórica")
plt.legend()
plt.grid(True)
plt.show()

# 3) Trayectorias con vectores de aceleración
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

#Teórica
axes[0].plot(x_teorico, y_teorico, "g--", label="Trayectoria teórica")
axes[0].quiver(x_teorico[::10], y_teorico[::10],
               np.zeros_like(x_teorico[::10]), -g*np.ones_like(y_teorico[::10]),
               color="blue", angles="xy", scale_units="xy", scale=150, width=0.004,
               label="Aceleración teórica")
axes[0].set_xlabel("X (cm)")
axes[0].set_ylabel("Altura Y (cm)")
axes[0].set_title("Trayectoria con aceleración teórica")
axes[0].legend()
axes[0].grid(True)

# Experimental
axes[1].plot(x_suav, y_suav, "o-", label="Trayectoria obtenida")
axes[1].quiver(x_suav[::2], y_suav[::2], ax[::2], ay[::2],
               color="red", angles="xy", scale_units="xy", scale=150, width=0.004,
               label="Aceleración obtenida")
axes[1].set_xlabel("X (cm)")
axes[1].set_ylabel("Altura Y (cm)")
axes[1].set_title("Trayectoria con aceleración obtenida")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()