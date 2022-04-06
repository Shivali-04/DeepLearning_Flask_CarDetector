#coding = utf-8
import sys
import os
import glob
import re
import numpy as np

from flask import Flask, render_template, request
# from keras.layers import Dense
from tensorflow.python.keras.layers import Dense
import tensorflow.python.keras
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import load_model
from werkzeug.utils import secure_filename
from tensorflow import tools
from tensorflow.tools import docs

app= Flask(__name__)

model= load_model('model_resnet50.h5')

def model_predict(img_path, model):
    img= image.load_img(img_path, target_size=(224,224))
    x= image.img_to_array(img)
    x= x/255
    x= np.expand_dims(x, axis=0)

    preds= model.predict(x)
    preds= np.argmax(preds, axis=1)

    if preds==0:
        preds="The Car IS Audi"
    elif preds==1:
        preds="The Car is Lamborghini"
    else:
        preds="The Car Is Mercedes"

    return preds



@app.route('/', methods= ['GET'])
def index():
    #main page
    return render_template('index.html')


@app.route('/predict', methods= ['POST'])
def upload():
    if request.method == 'POST':
        #get file from request
        f= request.files['file']

        #save the file to ./uploads
        basepath= os.path.dirname(__file__)
        file_path= os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)

        #Make Prediction
        result= model_predict(file_path,model)
        return result
    else: return None


if __name__  == '__main__':
    app.run(debug=True)












