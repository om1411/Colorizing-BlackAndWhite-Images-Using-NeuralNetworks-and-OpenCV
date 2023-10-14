import numpy as np
import cv2
from flask import Flask, render_template, request, send_file
import tempfile
import os

app = Flask(__name__)

# Define paths to model files and the temporary directory
prototxt_path = 'models/colorization_deploy_v2.prototxt'
model_path = 'models/colorization_release_v2.caffemodel'
kernel_path = 'models/pts_in_hull.npy'
temp_directory = 'temp/'

def colorize_image(image_path):
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    points = np.load(kernel_path)

    points = points.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    bw_image = cv2.imread(image_path)
    normalized = bw_image.astype("float32") / 255.0
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (bw_image.shape[1], bw_image.shape[0]))
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = (255.0 * colorized).astype("uint8")

    return colorized

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/colorize', methods=['POST'])
def colorize():
    # Check if the POST request has a file attached
    if 'image' not in request.files:
        return "No file uploaded"

    # Get the uploaded file
    uploaded_file = request.files['image']

    # Check if the file has a valid format (e.g., image)
    if uploaded_file.filename == '':
        return "No file selected"

    # Create a temporary directory to save the uploaded file
    os.makedirs(temp_directory, exist_ok=True)
    temp_path = os.path.join(temp_directory, 'temp_image.png')
    uploaded_file.save(temp_path)

    # Perform colorization on the uploaded image
    colorized = colorize_image(temp_path)

    # Save the colorized image to a temporary file
    temp_filename = os.path.join(temp_directory, 'colorized_image.jpg')
    cv2.imwrite(temp_filename, colorized)

    # Return the colorized image file to the client
    return send_file(temp_filename, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
