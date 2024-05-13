import numpy as np
import cv2

# Leer las imágenes
im1 = cv2.imread('img2.jpg')
ancho = int(im1.shape[1]/2)
alto = int(im1.shape[0]/2)
im1 = cv2.resize(im1, (ancho, alto), interpolation=cv2.INTER_AREA)

frame = cv2.imread('img1.jpg')
ancho = int(frame.shape[1])
alto = int(frame.shape[0])
frame = cv2.resize(frame, (ancho, alto), interpolation=cv2.INTER_AREA)

# Convertir las imágenes a escala de grises
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray_im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

# Detector de esquinas de Shi-Tomasi
corners_frame = cv2.goodFeaturesToTrack(gray_frame, maxCorners=500, qualityLevel=0.01, minDistance=10)
corners_im1 = cv2.goodFeaturesToTrack(gray_im1, maxCorners=500, qualityLevel=0.01, minDistance=10)

# Convertir las esquinas a puntos clave
keypoint1 = [cv2.KeyPoint(x[0][0], x[0][1], 10) for x in corners_im1]
keypoint2 = [cv2.KeyPoint(x[0][0], x[0][1], 10) for x in corners_frame]

# Descriptor BRISK
brisk = cv2.BRISK_create()
descriptor1 = brisk.compute(gray_im1, keypoint1)[1]
descriptor2 = brisk.compute(gray_frame, keypoint2)[1]

# 1. Creamos un objeto comparador de descriptores
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = matcher.match(descriptor1, descriptor2)

# 2. Ordenamos la lista
matches = sorted(matches, key=lambda x: x.distance)

# 3. Filtramos los resultados
goodmatches = int(len(matches) * 0.1)
matches = matches[:goodmatches]

# 4. Mostramos las coincidencias
img_matches = cv2.drawMatches(im1, keypoint1, frame, keypoint2, matches, None)

# Dibujar el número de coincidencias en la imagen de coincidencias
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img_matches, "Coincidencias: " + str(len(matches)), (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

# Mostrar un mensaje si hay más de 11 coincidencias
if 10 < len(matches) < 13:
    # Obtener el tamaño del texto
    text_size = cv2.getTextSize("Detectado", font, 1, 2)[0]

    # Calcular la posición centrada del texto
    text_x = (img_matches.shape[1] - text_size[0]) // 2
    text_y = (text_size[1]) + 10

    # Dibujar el texto centrado en la imagen de coincidencias
    cv2.putText(img_matches, "Detectado", (text_x, text_y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

# Mostrar caracteristicas
cv2.imshow("VIDEO CAPTURA", frame)
cv2.imshow("IMAGEN", im1)
cv2.imshow("COINCIDENCIAS", img_matches)

# Cerramos con lectura de teclado
cv2.waitKey(0)
cv2.destroyAllWindows()
