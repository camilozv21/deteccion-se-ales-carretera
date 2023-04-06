import cv2
import pickle
import itertools
import numpy as np
from sklearn import svm
from sklearn import tree
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions

# Load dataset
dataset = np.loadtxt('PROYECTO_FINAL\DataSet.txt', delimiter=',')
np.random.shuffle(dataset)
data, labels = dataset[:, 0:10], dataset[:, 10]

# Split dataset
data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.20, random_state=42)

features = []

# Train classifier
# K Vecino mas cercano
clfNN = KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree')  # Clasificador
clfTNN = clfNN.fit(data_train, labels_train)  # Entrenamiento
testNN = clfNN.score(data_test, labels_test)  # Prueba
print('Puntuacion K Vecino mas cercano:', testNN)  # Puntaje

# Start camera
camera = cv2.VideoCapture('PROYECTO_FINAL/videofinal.mp4')

camera.set(cv2.CAP_PROP_FPS, 10)
fps = int(camera.get(5))

while True:

    # Capture frame
    ret, img = camera.read()
    
    # imagen en grises
    imggg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # aplicamos filtros y umbral
    imgm = cv2.medianBlur(imggg, 9)
    _,imgB1=cv2.threshold(imgm,209,255,cv2.THRESH_BINARY)

    
    cv2.imshow('mask', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
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
    
    if max(lista_areas) < 10000:
        continue

    # Representamos el contorno más grande
    area = cv2.contourArea(mas_grande)
    x,y,w,h = cv2.boundingRect(mas_grande)
    #cv2.rectangle(opening1, (x,y), (x+w, y+h), (255,255,255), 2)

    # obtenemos las dimension
    H,W=opening1.shape[:2]
    imgArea = np.zeros((h, w), np.uint8)

    # extraemos lo que necesitamos de la imagen 
    imgArea[:] = opening1[y:y+h, x:x+w]
    hig,wid = imgArea.shape[:2]

    # extraccion de los momentos de la imagen junto con el area y la mitad de la imagen dividida en dos
    humm = cv2.HuMoments(cv2.moments(imgArea)).flatten()
    prom = np.mean(imgArea, dtype=np.float32)
    s1 = imgArea[0:hig, 0:wid//2]
    s2 = imgArea[0:hig, wid//2:wid]
    proms1 = np.mean(s1, dtype=np.float32)
    proms2 = np.mean(s2, dtype=np.float32)

    # asignacion de etiquetas 
    features = np.array([humm[0], humm[1], humm[2], humm[3], humm[4], humm[5], humm[6], prom, proms1, proms2])

    features = features.reshape(1, -1)
    signal = clfTNN.predict(features)
    
    if signal == 1:
        print('Avanzar')
    elif signal == 2:
        print('Avanzar derecha')
    elif signal == 3:
        print('Avanzar izquierda')    
    elif signal == 4:
        print('Giro derecha')
    elif signal == 5:
        print('Giro izquierda')
    elif signal == 6:
        print('Paso Peatonal')
    elif signal == 7:
        print('señal stop')
    
    # Display
    
camera.release()


'''# Ploteo de regiones de decision

X = data_train[:, [2, 3]]
y = labels_train.astype(int)

clf1 = tree.DecisionTreeClassifier(max_depth=4)
clf2 = GaussianNB()
clf3 = KNeighborsClassifier(n_neighbors=7)
clf4 = svm.SVC(kernel='rbf',  gamma=0.7, C=1.0)
clf5 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 1000), random_state=1)

clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
clf4.fit(X, y)
clf5.fit(X, y)

gs = gridspec.GridSpec(2, 2)

fig = plt.figure(figsize=(10, 8))

labels = ['Arbol de decision', 'Naive Bayes', 'K Vecino mas cercano', 'SVM']

for clf, lab, grd in zip([clf1, clf2, clf3, clf4],
                         labels,
                         itertools.product([0, 1], repeat=2)):
    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    plt.title(lab)

fig2 = plt.figure(figsize=(4, 4))
clf5.fit(X, y)
fig2 = plot_decision_regions(X=X, y=y, clf=clf5, legend=2)
plt.title('Red neuronal artificial')
plt.show()'''
