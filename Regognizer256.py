import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# Load models
detectorFace = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = keras.models.load_model("faceexpressiondetector256.h5")


# Set parameters
largura, altura = 256, 256
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0)



while (True):
    conectado, imagem = camera.read()
    imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    facesDetectadas = detectorFace.detectMultiScale(imagemCinza,
                                                    scaleFactor=1.5,
                                                    minSize=(30,30))
    for (x, y, l, a) in facesDetectadas:
        cv2.rectangle(imagem, (x, y), (x + l, y + a), (0,255,0), 2)

        photo = cv2.resize(imagemCinza[y:y + a, x:x + l], (largura, altura))
        photo=img_to_array(photo)
        photoarray = np.array(photo)
        photoarray = photoarray.reshape(1,256,256,1)
        photoarray=photoarray/255

        predict = model.predict(photoarray)
        maximo = np.argmax(predict, axis=1)


        nome= ""
        if maximo == 0:
            nome = 'Surpresa'
        elif maximo == 1:
            nome = 'Medo'
        elif maximo == 2:
            nome = 'Raiva'
        elif maximo == 4:
            nome = 'Tristesa'        
        elif maximo == 5:
            nome = 'Desgosto'
        elif maximo == 6:
            nome = 'Felicidade'
        else:
            nome = 'Neutro'
              
        cv2.putText(imagem, nome, (x,y +(a+30)), font, 2, (0,255,0))


    cv2.imshow("Face", imagem)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()