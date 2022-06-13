from flask import Flask, render_template, request
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

from keras.preprocessing import image
from keras.models import load_model

app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
    return render_template( "index.html")

@app.route("/", methods=['POST'])
def predict():
    reye = cv2.CascadeClassifier('haar_cascade/haarcascade_righteye_2splits.xml')
    model = load_model('models/eyedet.h5')

    imagefile = request.files['imagefile']
    path = "./images/" + imagefile.filename
    imagefile.save(path)

    image = cv2.imread(path)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    right_eye = reye.detectMultiScale(gray_img)

    for (x,y,w,h) in right_eye:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        eye_output = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(eye_output)

        r_eye=image[y:y+h,x:x+w]
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(128, 128))
        r_eye = r_eye/255
        r_eye = r_eye.reshape(128, 128, -1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict(r_eye)
        if rpred[0]<0.5:
            #classification = '%s ' % ()
            print(imagefile.filename + " ini adalah tertutup")
        else:
            #classification = '%s ' % ()
            print(imagefile.filename + " ini adalah terbuka")
        break
        classification = '%s ' % ()

    return render_template( "index.html")

if __name__ == '__main__':
    app.run(debug=True)