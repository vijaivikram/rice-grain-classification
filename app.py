from flask import Flask, render_template, request
from tensorflow import keras
from keras.models import load_model
import numpy as np
from keras.utils import load_img
import os
from keras.preprocessing import image
from keras.utils import img_to_array


app = Flask(__name__)
model = load_model('model.h5')
target_img = os.path.join(os.getcwd() , 'static/images')


@app.route('/')
def index_view():
    return render_template('index.html')
#Allow files with extension png, jpg and jpeg
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           

# Function to load and prepare the image in right shape
def read_image(filename):
    img = load_img(filename, target_size=(28, 28))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x


@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename): #Checking file format
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = read_image(file_path) #prepressing method
            class_prediction=model.predict(img) 
            classes_x=np.argmax(class_prediction,axis=1)
            if classes_x == 0:
              rice = "Arborio"
            elif classes_x == 1:
              rice = "Basmati"
            elif classes_x == 2:
              rice = "Ipsala"
            elif classes_x == 3:
              rice = "Jasmine"
            else:
              rice = "Karcadag"
            #'fruit' , 'prob' . 'user_image' these names we have seen in predict.html.
            return render_template('prediction.html', rice = rice,prob=class_prediction, user_image = file_path)
        else:
            return "Unable to read the file. Please check file extension"
if __name__ == '__main__':
    app.run(debug=True)