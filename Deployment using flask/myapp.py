# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 18:18:55 2020

@author: sahoo
"""
from __future__ import division, print_function

import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

model = load_model("C:/Users/sahoo/Recipes/saved_model/BananaLeaf_classifier.h5")
        # Necessary
# print('Model loaded. Start serving...')




def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(100,100))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    #x= x/255.0
    y_pred=model.predict(x.reshape(1,100,100,3))
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    preds = y_pred
    str1=''
    result=np.argmax(preds,axis=1) #take the index value of that array which value is maximum
    if result==0:
       str1='It has a disease called Black Bacterial Wilt'
    elif result==1:
       str1='It has a disease called Black Sigatoka Disease'
    elif result==2:
       str1='Wohh!!! It is a healthy Leaf'
    else:
       str1="It's not a banana leaf"
       
    return str1


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)


        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)