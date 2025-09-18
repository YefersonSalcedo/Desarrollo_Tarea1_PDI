import cv2
import os

# Ruta del video
video_path = "../../Videos Originales/V1.mp4"

# Carpeta donde se guardarán los frames
output_folder = "frames"
os.makedirs(output_folder, exist_ok=True)

# Cargar el video
cap = cv2.VideoCapture(video_path)

frame_number = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Si no hay más frames, termina el bucle

    # Guardar cada frame como imagen
    frame_name = f"{output_folder}/frame_{frame_number:04d}.jpg"
    cv2.imwrite(frame_name, frame)

    print(f"Guardado: {frame_name}")
    frame_number += 1

cap.release()
cv2.destroyAllWindows()