from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
import logging

from model import load_model

app = Flask(__name__)

# Define the path for the uploaded images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model once when the app starts
classifier = load_model("./check_points/model.keras")

@app.route('/')
def login():
    return render_template("index.html")

@app.route('/image/', methods=['POST'])
def display_label():
    # Check if a file is in the request
    if 'file' not in request.files:
        return "No file uploaded", 400
    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Save the file to the specified folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocess the image for prediction
    # Assuming the model expects images to be preprocessed into the appropriate format
    img = preprocess_image(file_path)

    # set up declaration of name
    dic = {0 : "Cat", 1 : "Dog"}

    # Make a prediction
    label = classifier.predict(img)

    print(label)

    # Convert predictions to a more readable format
    predicted_label = (label[0][0] > 0.5).astype(int)  # Assuming threshold of 0.5 for binary classification

    # Pass the file path and label to the label.html template
    return render_template("label.html", img=file.filename, label=dic[predicted_label])

def preprocess_image(file_path):
    """ 
    Preprocess the image for model prediction. 
    This is a placeholder function; you should implement the actual preprocessing as required by your model.
    """

    img = image.load_img(file_path, target_size=(256, 256))  # Adjust target size as per your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize if needed
    return img_array

if __name__ == "__main__":
    # Create the upload folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    logging.basicConfig(level=logging.INFO)
    app.run(debug=True)
