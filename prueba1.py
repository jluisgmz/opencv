import cv2

#creamos la video captura

cap=cv2.VideoCapture(0)

#creamos un ciclo para ejecutar nuestros Frames

while True:
    #leemos los fotogramas
    ret, frame = cap.read()

    print(ret)

    #mostrar los frames
    cv2.imshow("VIDEO CAPTURA", frame)

    #cerramos con lectura de teclado
    t = cv2.waitKey(1) #lee el teclado cada 1ms
    if t == 27: #codigo ascii para saber que numero representa una tecla (del teclado xd)
        break

#liberamos la VideoCaptura
cap.release()

#cerramos la ventana
cv2.destroyAllWindows()
