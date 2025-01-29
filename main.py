import cv2
import numpy as np
from utils.face_utils import detect_faces, recognize_faces

def main():
    # Inicializa a webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Captura frame por frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Detecta faces no frame
        faces = detect_faces(frame)
        
        # Reconhece as faces detectadas
        frame = recognize_faces(frame, faces)
        
        # Exibe o frame resultante
        cv2.imshow('Reconhecimento Facial', frame)
        
        # Pressione 'q' para sair do loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Libera a captura e fecha as janelas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()