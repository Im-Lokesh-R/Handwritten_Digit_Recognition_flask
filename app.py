from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the pre-trained model from uploads folder
MODEL_PATH = os.path.join(UPLOAD_FOLDER, 'mnist_digit_recognition_model.h5')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Ensure it is uploaded inside 'uploads/' directory.")

model = keras.models.load_model(MODEL_PATH)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Process image
            img = Image.open(filepath).convert('L')  # Convert to grayscale
            img = img.resize((28, 28))  # Resize to 28x28
            img_array = np.array(img) / 255.0  # Normalize pixel values
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension

            # Predict digit
            predictions = model.predict(img_array)
            predicted_digit = np.argmax(predictions)

            # Send result to frontend
            return render_template('index.html', prediction=predicted_digit, image_path=url_for('uploaded_file', filename=file.filename))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=PORT)
