import cv2

#creamos la video captura

cap=cv2.VideoCapture(0)

ancho = int(cap.get(3))
alto = int(cap.get(4))


print(ancho,alto)
# 'nombre de variable' = cv2.VideoWrite ('nombre', codificacion, FPS, Tamaño)
out = cv2.VideoWriter('Video.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 60, (ancho,alto))

#creamos un ciclo para ejecutar nuestros Frames

while True:
    #leemos los fotogramas
    ret, frame = cap.read()

    #guardamos el video
    out.write(frame)

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
