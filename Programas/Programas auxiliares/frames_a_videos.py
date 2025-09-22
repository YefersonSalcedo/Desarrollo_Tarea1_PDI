import cv2
import os

# Carpeta donde están los frames procesados
input_dir = "frames"

# Nombre del archivo de salida (video reconstruido en MP4)
output_video = "../Videos Procesados/V3_procesado.mp4"

# Obtener lista de archivos ordenados (aseguramos que sean imágenes)
frames = sorted([f for f in os.listdir(input_dir) if f.endswith(".jpg") or f.endswith(".png")])

if not frames:
    print("No se encontraron frames en la carpeta de salida.")
    exit()

# Leer el primer frame para obtener dimensiones
first_frame = cv2.imread(os.path.join(input_dir, frames[0]))
height, width, layers = first_frame.shape

# Definir codec y crear el objeto VideoWriter para MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec MP4
fps = 120
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Escribir todos los frames en el video
for frame_name in frames:
    frame_path = os.path.join(input_dir, frame_name)
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"No se pudo leer {frame_name}, lo salto.")
        continue
    out.write(frame)

# Liberar el objeto VideoWriter
out.release()
print(f"Video reconstruido en formato MP4 guardado en: {output_video}")
