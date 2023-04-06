import numpy as np
import cv2
import csv
import os

# inicializamos listas
chart = []
features = []

# Cargamos la imagen
# img = cv2.imread('D:/UNIVERSIDAD/artificial_vision/Imagenes_Vicion/Avanzar izquierda/dataset (610).png',0)

# path de la carpeta
input_images_path = "D:/UNIVERSIDAD/artificial_vision/Imagenes_Vicion/Stop"
files_names = os.listdir(input_images_path)

cont = 0 # Conteo de la imagen actual

# extraccion de las imagenes en la carpera 
for i in files_names:
    image_path = input_images_path + "/" + i
    print(image_path)
    img = cv2.imread(image_path,0)
    cont = cont + 1
    print(cont)
    if img is None:
        continue

    # aplicamos filtros y umbral
    imgm = cv2.medianBlur(img, 9)
    _,imgB1=cv2.threshold(imgm,209,255,cv2.THRESH_BINARY)

    # eliminamos ruido de la imagen
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3,3))
    opening1 = cv2.morphologyEx(imgB1, cv2.MORPH_CLOSE, kernel1, iterations=18)

    # hallamos los contornos de la imagen
    cnts,_ = cv2.findContours(opening1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Buscamos el contorno más grande
    lista_areas = []
    for c in cnts:
        area = cv2.contourArea(c)
        lista_areas.append(area)

    # Guardamos el area más grande
    mas_grande = cnts[lista_areas.index(max(lista_areas))]

    # Representamos el contorno más grande
    area = cv2.contourArea(mas_grande)
    x,y,w,h = cv2.boundingRect(mas_grande)
    #cv2.rectangle(opening1, (x,y), (x+w, y+h), (255,255,255), 2)

    # obtenemos las dimension
    H,W=opening1.shape[:2]
    imgArea = np.zeros((h, w), np.uint8)

    # extraemos lo que necesitamos de la imagen 
    imgArea[:] = opening1[y:y+h, x:x+w]
    cv2.imshow('binarizada', imgB1)
    cv2.imshow('img grande', imgArea)
    cv2.waitKey(0)
    hig,wid = imgArea.shape[:2]

    # extraccion de los momentos de la imagen junto con el area y la mitad de la imagen dividida en dos
    humm = cv2.HuMoments(cv2.moments(imgArea)).flatten()
    prom = np.mean(imgArea, dtype=np.float32)
    s1 = imgArea[0:hig, 0:wid//2]
    s2 = imgArea[0:hig, wid//2:wid]
    proms1 = np.mean(s1, dtype=np.float32)
    proms2 = np.mean(s2, dtype=np.float32)

    # asignacion de etiquetas 
    etiqueta = int(input('ingrtrese la etiqueta: '))
    chart = [humm[0], humm[1], humm[2], humm[3], humm[4], humm[5], humm[6], prom, proms1, proms2, etiqueta]
    features.append(chart)
    print(features)
    
    # creacion del archivo csv y guardado de los datos
    test = open("./hehehe.csv", 'w')
    wr = csv.writer(test, dialect='excel')
    for item in features:
        wr.writerow(item)

    cv2.destroyAllWindows()