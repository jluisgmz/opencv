# Importamos librerias
import numpy as np
import cv2

# Creamos la Video Captura
cap = cv2.VideoCapture(0)

# Leer la imagen original
im1 = cv2.imread('stop.png')
ancho = int(im1.shape[1])
alto = int(im1.shape[0])
im1 = cv2.resize(im1, (ancho, alto), interpolation = cv2.INTER_AREA)


frame = cv2.imread('bus.png')
alto_frame = frame.shape[0]

if alto_frame < 150:
    frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
elif alto_frame > 1500:
    frame = cv2.resize(frame, None, fx=0.16, fy=0.16, interpolation=cv2.INTER_AREA)
elif 900< alto_frame < 1500:
    frame = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
elif 700 < alto_frame < 900:
    frame = cv2.resize(frame, None, fx=0.33, fy=0.33, interpolation=cv2.INTER_AREA)
elif 500< alto_frame < 700:
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

# Creamos un ciclo para ejecutar nuestros Frames
while True:

    # Convertimos a EDG
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    # Buscamos puntos claves
    # Numero de puntos clave
    num_kpt = 500
    # Declaramos el objeto
    orb = cv2.ORB_create(num_kpt)
    # Extraemos la info de la img
    keypoint1, descriptor1 = orb.detectAndCompute(gray_im1, None)
    # Extraemos la info de los frames
    keypoint2, descriptor2 = orb.detectAndCompute(gray_frame, None)

    print(descriptor1)


    # Dibujamos
    im1_display = cv2.drawKeypoints(im1, keypoint1, outImage = np.array([]), color =(255,0,0),
                                    flags= cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    frame_display = cv2.drawKeypoints(frame, keypoint2, outImage=np.array([]), color=(255, 0, 0),
                                    flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

    # ¿Como hacemos coincidir los puntos?
    # 1. Creamos un objeto comparador de descriptores
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = matcher.match(descriptor1, descriptor2)

    # 2. Ordenamos la lista
    matches = sorted(matches, key = lambda x: x.distance, reverse = False)

    # 3. Filtramos los resultados
    goodmatches = int(len(matches) * 0.1)
    matches = matches[:goodmatches]

    # 4. Mostramos las coincidencias
    img_matches = cv2.drawMatches(im1, keypoint1, frame, keypoint2, matches, None)


    # Dibujar el número de coincidencias en la imagen de coincidencias
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_matches, "Coincidencias: " + str(len(matches)), (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar un mensaje si hay más de 11 coincidencias
    if len(matches) > 7:
        # Obtener el tamaño del texto
        text_size = cv2.getTextSize("Detectado", font, 1, 2)[0]

        # Calcular la posición centrada del texto
        text_x = (text_size[0]) + 170
        text_y = (text_size[1]) + 10

        # Dibujar el texto centrado en la imagen de coincidencias
        cv2.putText(img_matches, "Detectado", (text_x, text_y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostramos caracteristicas
    cv2.imshow("VIDEO CAPTURA", frame_display)
    cv2.imshow("IMAGEN", im1_display)
    cv2.imshow("COINCIDENCIAS", img_matches)

    # Cerramos con lectura de teclado
    t = cv2.waitKey(1)
    if t == 27:
        break

# Liberamos la VideoCaptura
cap.release()
# Cerramos la ventana
cv2.destroyAllWindows()