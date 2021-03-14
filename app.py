# -*- coding: utf-8 -*-


from flask import Flask, render_template, request, redirect, url_for, flash
import os
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
UPLOAD_FOLDER = './flask app/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
# Create Database if it doesnt exist

app = Flask(__name__,static_url_path='/assets',
            static_folder='./flask app/assets', 
            template_folder='./flask app')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def root():
   return render_template('index.html')

@app.route('/index.html')
def index():
   return render_template('index.html')

@app.route('/upload.html')
def upload():
   return render_template('upload.html')

@app.route('/upload_chest.html')
def upload_chest():
   return render_template('upload_chest.html')

@app.route('/uploaded_chest', methods = ['POST', 'GET'])
def uploaded_chest():
   if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_chest.jpg'))

   model= load_model('models/model.hdf5')
   test_image = image.load_img('./flask app/assets/images/upload_chest.jpg',target_size=(224, 224))
   test_image = image.img_to_array(test_image)
   test_image = np.expand_dims(test_image, axis=0)
   test_image = test_image.reshape(1, 224, 224, 3)  # Ambiguity!
   pred = model.predict(test_image, batch_size=1)
   probability = pred[0]
   print("Predictions:")
   if probability[0] > 0.7:
      model_pred = str('%.2f' % (probability[0]*100) + '% Yes')
   else:
     model_pred = str('%.2f' % ((1-probability[0])*100) + '% No')
   print(model_pred)
   return render_template('index.html',model_pred=model_pred)
if __name__ == '__main__':
   app.secret_key = ".."
   app.run()
