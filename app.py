from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model (make sure to replace 'model.h5' with the path to your model file)
model = keras.models.load_model('model.h5')

app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        
        # If no file is selected
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Process the image for prediction
            img = Image.open(filename).convert('L')  # Convert to grayscale
            img = img.resize((28, 28))  # Resize to 28x28
            img_array = np.array(img) / 255.0  # Normalize the pixel values
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension (1 for grayscale)

            # Predict the digit
            predictions = model.predict(img_array)
            predicted_digit = np.argmax(predictions)  # Get the class with the highest probability

            # Return the result to the user
            return render_template('index.html', prediction=predicted_digit, image_path=filename)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
