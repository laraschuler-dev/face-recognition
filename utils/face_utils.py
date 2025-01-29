import cv2
import os

def detect_faces(image):
    # Caminho relativo para o arquivo Haar Cascade
    cascade_path = os.path.join("data", "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Verifica se o arquivo foi carregado corretamente
    if face_cascade.empty():
        raise Exception("Erro ao carregar o classificador Haar Cascade. Verifique o caminho do arquivo.")
    
    # Converte a imagem para escala de cinza
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detecta faces na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return faces

def recognize_faces(image, faces):
    # desenha retângulos ao redor das faces detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return image