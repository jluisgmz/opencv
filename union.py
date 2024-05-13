import numpy as np
import cv2
import sys



def salir():
    
    # Cerramos la ventana
    cv2.destroyAllWindows()
    sys.exit()

#señales informativas

def bus():
    # Creamos la Video Captura
    cap = cv2.VideoCapture(0)

    # Leer la imagen original
    im1 = cv2.imread('bus.png')
    ancho = int(im1.shape[1]/6)
    alto = int(im1.shape[0]/6)
    im1 = cv2.resize(im1, (ancho, alto), interpolation=cv2.INTER_AREA)

    frame = imagen_a_comparar
    alto_frame = frame.shape[0]

    if alto_frame < 150:
        frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
    elif 150< alto_frame < 340:
        frame = cv2.resize(frame, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_AREA)
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

        # Dibujamos
        im1_display = cv2.drawKeypoints(im1, keypoint1, outImage=np.array([]), color=(255, 0, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        frame_display = cv2.drawKeypoints(frame, keypoint2, outImage=np.array([]), color=(255, 0, 0),
                                           flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        # ¿Como hacemos coincidir los puntos?
        # 1. Creamos un objeto comparador de descriptores
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptor1, descriptor2)

        # 2. Ordenamos la lista
        matches = sorted(matches, key=lambda x: x.distance, reverse=False)

        # 3. Filtramos los resultados
        goodmatches = int(len(matches) * 0.1)
        matches = matches[:goodmatches]

        # 4. Mostramos las coincidencias
        img_matches = cv2.drawMatches(im1, keypoint1, frame, keypoint2, matches, None)

        # Dibujar el número de coincidencias en la imagen de coincidencias
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_matches, "Coincidencias: " + str(len(matches)), (10, 30), font, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        # Mostrar un mensaje si hay más de 11 coincidencias
        if len(matches) > 15:
            # Obtener el tamaño del texto
            text_size = cv2.getTextSize("****  **", font, 1, 2)[0]
            text_size2 = cv2.getTextSize("********", font, 1, 2)[0]

            # Calcular la posición centrada del texto
            text_x = (text_size[0])+150
            text_y = (text_size[1])+5

            text_x2 = (text_size2[0])+150
            text_y2 = (text_size2[1]) + 25

            # Dibujar el texto centrado en la imagen de coincidencias
            cv2.putText(img_matches, "Parada", (text_x, text_y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img_matches, "de Bus", (text_x2, text_y2), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Mostramos características
            cv2.imshow("VIDEO CAPTURA", frame_display)
            cv2.imshow("IMAGEN", im1_display)
            cv2.imshow("COINCIDENCIAS", img_matches)

        else:
            silla()

        # Cerramos con lectura de teclado
        t = cv2.waitKey(1)
        if t == 27:
            break

    # Liberamos la VideoCaptura
    cap.release()
    # Cerramos la ventana
    cv2.destroyAllWindows()
    salir()
    # Creamos la Video Captura
    cap = cv2.VideoCapture(0)

    # Leer la imagen original
    im1 = cv2.imread('bus.png')
    ancho = int(im1.shape[1]/6)
    alto = int(im1.shape[0]/6)
    im1 = cv2.resize(im1, (ancho, alto), interpolation=cv2.INTER_AREA)

    frame = imagen_a_comparar
    alto_frame = frame.shape[0]

    if alto_frame < 150:
        frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
    elif 150< alto_frame < 340:
        frame = cv2.resize(frame, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_AREA)
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

        # Dibujamos
        im1_display = cv2.drawKeypoints(im1, keypoint1, outImage=np.array([]), color=(255, 0, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        frame_display = cv2.drawKeypoints(frame, keypoint2, outImage=np.array([]), color=(255, 0, 0),
                                           flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        # ¿Como hacemos coincidir los puntos?
        # 1. Creamos un objeto comparador de descriptores
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptor1, descriptor2)

        # 2. Ordenamos la lista
        matches = sorted(matches, key=lambda x: x.distance, reverse=False)

        # 3. Filtramos los resultados
        goodmatches = int(len(matches) * 0.1)
        matches = matches[:goodmatches]

        # 4. Mostramos las coincidencias
        img_matches = cv2.drawMatches(im1, keypoint1, frame, keypoint2, matches, None)

        # Dibujar el número de coincidencias en la imagen de coincidencias
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_matches, "Coincidencias: " + str(len(matches)), (10, 30), font, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        # Mostrar un mensaje si hay más de 11 coincidencias
        if len(matches) > 8:
            # Obtener el tamaño del texto
            text_size = cv2.getTextSize("Detectado", font, 1, 2)[0]

            # Calcular la posición centrada del texto
            text_x = (text_size[0]) + 170
            text_y = (text_size[1]) + 10

            # Dibujar el texto centrado en la imagen de coincidencias
            cv2.putText(img_matches, "Detectado", (text_x, text_y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Mostramos características
            cv2.imshow("VIDEO CAPTURA", frame_display)
            cv2.imshow("IMAGEN", im1_display)
            cv2.imshow("COINCIDENCIAS", img_matches)

        else:
            silla()

        # Cerramos con lectura de teclado
        t = cv2.waitKey(1)
        if t == 27:
            break

    # Liberamos la VideoCaptura
    cap.release()
    # Cerramos la ventana
    cv2.destroyAllWindows()
    salir()

def silla():
    # Creamos la Video Captura
    cap = cv2.VideoCapture(0)

    # Leer la imagen original
    im1 = cv2.imread('silla.jpg')
    ancho = int(im1.shape[1])
    alto = int(im1.shape[0])
    im1 = cv2.resize(im1, (ancho, alto), interpolation=cv2.INTER_AREA)

    frame = imagen_a_comparar
    alto_frame = frame.shape[0]

    if alto_frame < 175:
        frame = cv2.resize(frame, None, fx=2.4, fy=2.4, interpolation=cv2.INTER_AREA)
    elif 175< alto_frame < 220:
        frame = cv2.resize(frame, None, fx=1.4, fy=1.4, interpolation=cv2.INTER_AREA)
    elif 290< alto_frame < 500:
        frame = cv2.resize(frame, None, fx=0.55, fy=0.55, interpolation=cv2.INTER_AREA)
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

        # Dibujamos
        im1_display = cv2.drawKeypoints(im1, keypoint1, outImage=np.array([]), color=(255, 0, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        frame_display = cv2.drawKeypoints(frame, keypoint2, outImage=np.array([]), color=(255, 0, 0),
                                           flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        # ¿Como hacemos coincidir los puntos?
        # 1. Creamos un objeto comparador de descriptores
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptor1, descriptor2)

        # 2. Ordenamos la lista
        matches = sorted(matches, key=lambda x: x.distance, reverse=False)

        # 3. Filtramos los resultados
        goodmatches = int(len(matches) * 0.1)
        matches = matches[:goodmatches]

        # 4. Mostramos las coincidencias
        img_matches = cv2.drawMatches(im1, keypoint1, frame, keypoint2, matches, None)

        # Dibujar el número de coincidencias en la imagen de coincidencias
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_matches, "Coincidencias: " + str(len(matches)), (10, 30), font, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        # Mostrar un mensaje si hay más de 11 coincidencias
        if len(matches) > 4:
            # Obtener el tamaño del texto
            text_size = cv2.getTextSize("*******", font, 1, 2)[0]
            text_size2 = cv2.getTextSize("***********", font, 1, 2)[0]

            # Calcular la posición centrada del texto
            text_x = (text_size[0])+150
            text_y = (text_size[1])+170

            text_x2 = (text_size2[0])+50
            text_y2 = (text_size2[1]) + 195

            # Dibujar el texto centrado en la imagen de coincidencias
            cv2.putText(img_matches, "Parqueo", (text_x, text_y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img_matches, "Discapacitados", (text_x2, text_y2), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Mostramos características
            cv2.imshow("VIDEO CAPTURA", frame_display)
            cv2.imshow("IMAGEN", im1_display)
            cv2.imshow("COINCIDENCIAS", img_matches)

        else:
            # Mostrar mensaje de que la imagen no es reconocida
            img_no_match = np.zeros((200, 600, 3), dtype=np.uint8)  # Crea una imagen negra
            cv2.putText(img_no_match, "No reconocemos esta imagen", (50, 100), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("No Match", img_no_match)

        # Cerramos con lectura de teclado
        t = cv2.waitKey(1)
        if t == 27:
            break

    # Liberamos la VideoCaptura
    cap.release()
    # Cerramos la ventana
    cv2.destroyAllWindows()
    salir()

#señales preventivas

def curva():
    # Creamos la Video Captura
    cap = cv2.VideoCapture(0)

    # Leer la imagen original
    im1 = cv2.imread('curva.png')
    ancho = int(im1.shape[1])
    alto = int(im1.shape[0])
    im1 = cv2.resize(im1, (ancho, alto), interpolation=cv2.INTER_AREA)

    frame = imagen_a_comparar
    alto_frame = frame.shape[0]

    if alto_frame < 150:
        frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
    elif alto_frame > 900:
        frame = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    elif 700 < alto_frame < 900:
        frame = cv2.resize(frame, None, fx=0.33, fy=0.33, interpolation=cv2.INTER_AREA)
    elif 500 < alto_frame < 700:
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

        # Dibujamos
        im1_display = cv2.drawKeypoints(im1, keypoint1, outImage=np.array([]), color=(255, 0, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        frame_display = cv2.drawKeypoints(frame, keypoint2, outImage=np.array([]), color=(255, 0, 0),
                                           flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        # ¿Como hacemos coincidir los puntos?
        # 1. Creamos un objeto comparador de descriptores
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptor1, descriptor2)

        # 2. Ordenamos la lista
        matches = sorted(matches, key=lambda x: x.distance, reverse=False)

        # 3. Filtramos los resultados
        goodmatches = int(len(matches) * 0.1)
        matches = matches[:goodmatches]

        # 4. Mostramos las coincidencias
        img_matches = cv2.drawMatches(im1, keypoint1, frame, keypoint2, matches, None)

        # Dibujar el número de coincidencias en la imagen de coincidencias
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_matches, "Coincidencias: " + str(len(matches)), (10, 30), font, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        # Mostrar un mensaje si hay más de 11 coincidencias
        if len(matches) > 7:
            # Obtener el tamaño del texto
            text_size = cv2.getTextSize("Detectado", font, 1, 2)[0]

            # Calcular la posición centrada del texto
            text_x = (text_size[0]) + 170
            text_y = (text_size[1]) + 10

            # Dibujar el texto centrado en la imagen de coincidencias
            cv2.putText(img_matches, "Detectado", (text_x, text_y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Mostramos características
            cv2.imshow("VIDEO CAPTURA", frame_display)
            cv2.imshow("IMAGEN", im1_display)
            cv2.imshow("COINCIDENCIAS", img_matches)


        else:
            peatones()

        

        # Cerramos con lectura de teclado
        t = cv2.waitKey(1)
        if t == 27:
            break

    # Liberamos la VideoCaptura
    cap.release()
    # Cerramos la ventana
    cv2.destroyAllWindows()


def peatones():

    # Leer la imagen original
    im1 = cv2.imread('peatones.png')
    ancho = int(im1.shape[1])
    alto = int(im1.shape[0])
    im1 = cv2.resize(im1, (ancho, alto), interpolation=cv2.INTER_AREA)

    frame = imagen_a_comparar
    alto_frame = frame.shape[0]

    if alto_frame < 150:
        frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
    elif alto_frame > 900:
        frame = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    elif 700 < alto_frame < 900:
        frame = cv2.resize(frame, None, fx=0.33, fy=0.33, interpolation=cv2.INTER_AREA)
    elif 500 < alto_frame < 700:
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

        # Dibujamos
        im1_display = cv2.drawKeypoints(im1, keypoint1, outImage=np.array([]), color=(255, 0, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        frame_display = cv2.drawKeypoints(frame, keypoint2, outImage=np.array([]), color=(255, 0, 0),
                                           flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        # ¿Como hacemos coincidir los puntos?
        # 1. Creamos un objeto comparador de descriptores
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptor1, descriptor2)

        # 2. Ordenamos la lista
        matches = sorted(matches, key=lambda x: x.distance, reverse=False)

        # 3. Filtramos los resultados
        goodmatches = int(len(matches) * 0.1)
        matches = matches[:goodmatches]

        # 4. Mostramos las coincidencias
        img_matches = cv2.drawMatches(im1, keypoint1, frame, keypoint2, matches, None)

        # Dibujar el número de coincidencias en la imagen de coincidencias
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_matches, "Coincidencias: " + str(len(matches)), (10, 30), font, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        # Mostrar un mensaje si hay más de 11 coincidencias
        if len(matches) > 15:
            # Obtener el tamaño del texto
            text_size = cv2.getTextSize("Detectado", font, 1, 2)[0]

            # Calcular la posición centrada del texto
            text_x = (text_size[0]) + 170
            text_y = (text_size[1]) + 10

            # Dibujar el texto centrado en la imagen de coincidencias
            cv2.putText(img_matches, "Detectado", (text_x, text_y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Mostramos características
            cv2.imshow("VIDEO CAPTURA", frame_display)
            cv2.imshow("IMAGEN", im1_display)
            cv2.imshow("COINCIDENCIAS", img_matches)
        
        else:
            derrumbe()

        # Cerramos con lectura de teclado
        t = cv2.waitKey(1)
        if t == 27:
            break
            

    # Cerramos la ventana
    cv2.destroyAllWindows()
    salir()


def derrumbe():

    # Leer la imagen original
    im1 = cv2.imread('derrumbe.png')
    ancho = int(im1.shape[1])
    alto = int(im1.shape[0])
    im1 = cv2.resize(im1, (ancho, alto), interpolation=cv2.INTER_AREA)

    frame = imagen_a_comparar
    alto_frame = frame.shape[0]

    if alto_frame < 150:
        frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
    elif alto_frame > 900:
        frame = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    elif 700 < alto_frame < 900:
        frame = cv2.resize(frame, None, fx=0.33, fy=0.33, interpolation=cv2.INTER_AREA)
    elif 500 < alto_frame < 700:
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

        # Dibujamos
        im1_display = cv2.drawKeypoints(im1, keypoint1, outImage=np.array([]), color=(255, 0, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        frame_display = cv2.drawKeypoints(frame, keypoint2, outImage=np.array([]), color=(255, 0, 0),
                                           flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        # ¿Como hacemos coincidir los puntos?
        # 1. Creamos un objeto comparador de descriptores
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptor1, descriptor2)

        # 2. Ordenamos la lista
        matches = sorted(matches, key=lambda x: x.distance, reverse=False)

        # 3. Filtramos los resultados
        goodmatches = int(len(matches) * 0.1)
        matches = matches[:goodmatches]

        # 4. Mostramos las coincidencias
        img_matches = cv2.drawMatches(im1, keypoint1, frame, keypoint2, matches, None)

        # Dibujar el número de coincidencias en la imagen de coincidencias
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_matches, "Coincidencias: " + str(len(matches)), (10, 30), font, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        # Mostrar un mensaje si hay más de 11 coincidencias
        if len(matches) > 5:
            # Obtener el tamaño del texto
            text_size = cv2.getTextSize("****  **", font, 1, 2)[0]
            text_size2 = cv2.getTextSize("********", font, 1, 2)[0]

            # Calcular la posición centrada del texto
            text_x = (text_size[0])+150
            text_y = (text_size[1])+5

            text_x2 = (text_size2[0])+150
            text_y2 = (text_size2[1]) + 25

            # Dibujar el texto centrado en la imagen de coincidencias
            cv2.putText(img_matches, "Zona de", (text_x, text_y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img_matches, "Derrumbe", (text_x2, text_y2), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
             
            # Mostramos características
            cv2.imshow("VIDEO CAPTURA", frame_display)
            cv2.imshow("IMAGEN", im1_display)
            cv2.imshow("COINCIDENCIAS", img_matches)
        
        else:
            # Mostrar mensaje de que la imagen no es reconocida
            img_no_match = np.zeros((200, 600, 3), dtype=np.uint8)  # Crea una imagen negra
            cv2.putText(img_no_match, "No reconocemos esta imagen", (50, 100), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("No Match", img_no_match)  
        

        # Cerramos con lectura de teclado
        t = cv2.waitKey(1)
        if t == 27:
            break
            

    # Cerramos la ventana
    cv2.destroyAllWindows()
    salir()

#señales reglamentarias

def alto():
    # Creamos la Video Captura
    cap = cv2.VideoCapture(0)

    # Leer la imagen original
    im1 = cv2.imread('stop.png')
    ancho = int(im1.shape[1])
    alto = int(im1.shape[0])
    im1 = cv2.resize(im1, (ancho, alto), interpolation=cv2.INTER_AREA)

    frame = imagen_a_comparar
    alto_frame = frame.shape[0]

    if alto_frame < 150:
        frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
    elif alto_frame > 900:
        frame = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    elif 700 < alto_frame < 900:
        frame = cv2.resize(frame, None, fx=0.33, fy=0.33, interpolation=cv2.INTER_AREA)
    elif 500 < alto_frame < 700:
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

        # Dibujamos
        im1_display = cv2.drawKeypoints(im1, keypoint1, outImage=np.array([]), color=(255, 0, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        frame_display = cv2.drawKeypoints(frame, keypoint2, outImage=np.array([]), color=(255, 0, 0),
                                           flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        # ¿Como hacemos coincidir los puntos?
        # 1. Creamos un objeto comparador de descriptores
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptor1, descriptor2)

        # 2. Ordenamos la lista
        matches = sorted(matches, key=lambda x: x.distance, reverse=False)

        # 3. Filtramos los resultados
        goodmatches = int(len(matches) * 0.1)
        matches = matches[:goodmatches]

        # 4. Mostramos las coincidencias
        img_matches = cv2.drawMatches(im1, keypoint1, frame, keypoint2, matches, None)

        # Dibujar el número de coincidencias en la imagen de coincidencias
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_matches, "Coincidencias: " + str(len(matches)), (10, 30), font, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        # Mostrar un mensaje si hay más de 11 coincidencias
        if len(matches) > 7:
            # Obtener el tamaño del texto
            text_size = cv2.getTextSize("Detectado", font, 1, 2)[0]

            # Calcular la posición centrada del texto
            text_x = (text_size[0]) + 170
            text_y = (text_size[1]) + 10

            # Dibujar el texto centrado en la imagen de coincidencias
            cv2.putText(img_matches, "STOP", (text_x, text_y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            ceda()

        # Mostramos características
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


def ceda():

    # Leer la imagen original
    im1 = cv2.imread('ceda.png')
    ancho = int(im1.shape[1])
    alto = int(im1.shape[0])
    im1 = cv2.resize(im1, (ancho, alto), interpolation=cv2.INTER_AREA)

    frame = imagen_a_comparar
    alto_frame = frame.shape[0]

    if alto_frame < 150:
        frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
    elif alto_frame > 900:
        frame = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    elif 700 < alto_frame < 900:
        frame = cv2.resize(frame, None, fx=0.33, fy=0.33, interpolation=cv2.INTER_AREA)
    elif 500 < alto_frame < 700:
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

        # Dibujamos
        im1_display = cv2.drawKeypoints(im1, keypoint1, outImage=np.array([]), color=(255, 0, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        frame_display = cv2.drawKeypoints(frame, keypoint2, outImage=np.array([]), color=(255, 0, 0),
                                           flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        # ¿Como hacemos coincidir los puntos?
        # 1. Creamos un objeto comparador de descriptores
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptor1, descriptor2)

        # 2. Ordenamos la lista
        matches = sorted(matches, key=lambda x: x.distance, reverse=False)

        # 3. Filtramos los resultados
        goodmatches = int(len(matches) * 0.1)
        matches = matches[:goodmatches]

        # 4. Mostramos las coincidencias
        img_matches = cv2.drawMatches(im1, keypoint1, frame, keypoint2, matches, None)

        # Dibujar el número de coincidencias en la imagen de coincidencias
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_matches, "Coincidencias: " + str(len(matches)), (10, 30), font, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        # Mostrar un mensaje si hay más de 11 coincidencias
        if len(matches) > 9:
            # Obtener el tamaño del texto
            text_size = cv2.getTextSize("Detectado", font, 1, 2)[0]

            # Calcular la posición centrada del texto
            text_x = (text_size[0]) + 170
            text_y = (text_size[1]) + 10

            # Dibujar el texto centrado en la imagen de coincidencias
            cv2.putText(img_matches, "CEDA EL PASO", (text_x, text_y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        else:
            revasar()

        # Mostramos características
        cv2.imshow("VIDEO CAPTURA", frame_display)
        cv2.imshow("IMAGEN", im1_display)
        cv2.imshow("COINCIDENCIAS", img_matches)
        

        # Cerramos con lectura de teclado
        t = cv2.waitKey(1)
        if t == 27:
            break
            

    # Cerramos la ventana
    cv2.destroyAllWindows()
    salir()


def revasar():

    # Leer la imagen original
    im1 = cv2.imread('revasar.png')
    ancho = int(im1.shape[1])
    alto = int(im1.shape[0])
    im1 = cv2.resize(im1, (ancho, alto), interpolation=cv2.INTER_AREA)

    frame = imagen_a_comparar
    alto_frame = frame.shape[0]

    if alto_frame < 150:
        frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
    elif alto_frame > 900:
        frame = cv2.resize(frame, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    elif 700 < alto_frame < 900:
        frame = cv2.resize(frame, None, fx=0.33, fy=0.33, interpolation=cv2.INTER_AREA)
    elif 500 < alto_frame < 700:
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

        # Dibujamos
        im1_display = cv2.drawKeypoints(im1, keypoint1, outImage=np.array([]), color=(255, 0, 0),
                                        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        frame_display = cv2.drawKeypoints(frame, keypoint2, outImage=np.array([]), color=(255, 0, 0),
                                           flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

        # ¿Como hacemos coincidir los puntos?
        # 1. Creamos un objeto comparador de descriptores
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptor1, descriptor2)

        # 2. Ordenamos la lista
        matches = sorted(matches, key=lambda x: x.distance, reverse=False)

        # 3. Filtramos los resultados
        goodmatches = int(len(matches) * 0.1)
        matches = matches[:goodmatches]

        # 4. Mostramos las coincidencias
        img_matches = cv2.drawMatches(im1, keypoint1, frame, keypoint2, matches, None)

        # Dibujar el número de coincidencias en la imagen de coincidencias
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_matches, "Coincidencias: " + str(len(matches)), (10, 30), font, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)

        # Mostrar un mensaje si hay más de 11 coincidencias
        if len(matches) > 9:
            # Obtener el tamaño del texto
            text_size = cv2.getTextSize("Detectado", font, 1, 2)[0]

            # Calcular la posición centrada del texto
            text_x = (text_size[0]) + 170
            text_y = (text_size[1]) + 10

            # Dibujar el texto centrado en la imagen de coincidencias
            cv2.putText(img_matches, "NO REVASAR", (text_x, text_y), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Mostramos características
            cv2.imshow("VIDEO CAPTURA", frame_display)
            cv2.imshow("IMAGEN", im1_display)
            cv2.imshow("COINCIDENCIAS", img_matches)

        else:
            # Mostrar mensaje de que la imagen no es reconocida
            img_no_match = np.zeros((200, 600, 3), dtype=np.uint8)  # Crea una imagen negra
            cv2.putText(img_no_match, "No reconocemos esta imagen", (50, 100), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow("No Match", img_no_match)   

        
        

        # Cerramos con lectura de teclado
        t = cv2.waitKey(1)
        if t == 27:
            break
            

    # Cerramos la ventana
    cv2.destroyAllWindows()
    salir()


def detectar_color(imagen):
    # Convertir la imagen a espacio de color HSV
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    # Rangos de los colores en HSV
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Máscaras para cada color
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Contornos de cada máscara
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Área máxima de cada contorno
    area_blue = max([cv2.contourArea(c) for c in contours_blue], default=0)
    area_yellow = max([cv2.contourArea(c) for c in contours_yellow], default=0)
    area_red = max([cv2.contourArea(c) for c in contours_red], default=0)

    # Determinar el color predominante
    if area_blue > area_yellow and area_blue > area_red:
        print("La imagen a comparar pertenece a SEÑALES INFORMATIVAS.")
        bus()
    elif area_yellow > area_blue and area_yellow > area_red:
        print("La imagen a comparar pertenece a SEÑALES PREVENTIVAS.")
        curva()
    elif area_red > area_blue and area_red > area_yellow:
        print("La imagen a comparar pertenece a SEÑALES REGLAMENTARIAS.")
        alto()
    else:
        print("No se pudo determinar el tipo de señal de la imagen a comparar.")

# Cargar las imágenes
imagen_azul = cv2.imread('azul.jpg')
imagen_amarilla = cv2.imread('amarillo.jpg')
imagen_roja = cv2.imread('rojo.jpg')
#imagen a procesar
imagen_a_comparar = cv2.imread('silla2.png')

# Llamar a la función detectar_color para la imagen a comparar
detectar_color(imagen_a_comparar)
