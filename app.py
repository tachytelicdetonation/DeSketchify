from flask import Flask, render_template,  request, redirect, url_for, send_from_directory

import os
import cv2
import sys

from fastai.vision import *
from werkzeug.utils import secure_filename
from PIL import Image
from loss import FeatureLoss

UPLOAD_FOLDER = 'static/images/uploads' # folder where images are uploaded
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
INPUT_FILENAME = " "
OUTPUT_FILENAME = " "

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def reduce_dims(image_path):
    imageFile = image_path
    im1 = Image.open(imageFile)
    # adjust width and height to your needs
    width = 400
    height = 300
    im5 = im1.resize((width, height), Image.ANTIALIAS)    # best down-sizing filter
    ext = ".jpg"
    im5.save(image_path)


app.debug = True
@app.route('/', methods=['POST','GET'])
def index():
    image_files= []
    input_file_name = ' '

    if request.method == 'POST':

        file = request.files['file']

        if file.filename == '':
            print('No selected file')

        if file and allowed_file(file.filename):
            global INPUT_FILENAME
            INPUT_FILENAME = secure_filename(file.filename)
            print(INPUT_FILENAME)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], INPUT_FILENAME))

        input_file_name = '../' + os.path.join(app.config['UPLOAD_FOLDER'], INPUT_FILENAME)

    if request.method == 'GET':
        print("GET")
    return render_template ('index.html',  input_image = input_file_name) #This line will render files from the folder templates





@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/result', methods=['POST','GET'])
def result():

    if(INPUT_FILENAME == ' '):
        return 'Empty'

    reduce_dims(os.path.join(app.config['UPLOAD_FOLDER'], INPUT_FILENAME))
    
    input_image = open_image(os.path.join(app.config['UPLOAD_FOLDER'], INPUT_FILENAME))
    
    learn = load_learner('models/').to_fp16()
    generated_image = learn.predict(input_image)
    #generated_image = cv2.Canny(input_image, 100,200)
    #generated_image = generated_image[0]
    generated_image[0].save('./static/images/output/generated_image.jpg')


    if request.method == 'POST':
        result =  request.form
        return render_template('result.html', generated_image = './' + "static/images/output/generated_image.jpg")

    

if __name__ == '__main__':
    app.run()