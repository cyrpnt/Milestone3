from flask import Flask, request, render_template, send_from_directory
from PIL import Image
import numpy as np
import tensorflow as tf
import io
from os import path
from dnn_best import create_dnn

app = Flask(__name__)

# Load the DNN
dnn = './train_model.h5'
if path.exists(dnn) == False:
    create_dnn()
model = tf.keras.models.load_model('./train_model.h5')

UPLOAD_FOLDER = './'

@app.route('/', methods=['GET', 'POST'])
def process_image():
    if request.method == 'GET':
        # Template for the API
        return render_template('upload.html')
    
    if request.method == 'POST':
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))
        image = image.resize((64, 64)) # The best input shape we have had in the fine tuning phase
        image = image.convert("RGB") # Converts the image to another format
        image_array = np.array(image) / 255.0  # Normalisation

        # Redimmension
        image_array = np.expand_dims(image_array, axis=0)

        # Deblurring
        print("\n\n\nici\n\n\n")

        processed_image = model.predict(image_array)

        # Convert back to an image
        processed_image = np.squeeze(processed_image) * 255
        processed_image = processed_image.astype(np.uint8)
        img = Image.fromarray(processed_image)

        # Save and download to the user's computer the output
        img.save('output.png')
        return send_from_directory(UPLOAD_FOLDER, "output.png", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)