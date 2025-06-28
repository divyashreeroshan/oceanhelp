from flask import Flask, render_template, request, send_file, redirect, url_for
import cv2
import numpy as np
import os
import uuid
from PIL import Image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
PROCESSED_FOLDER = 'static/processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Resize large images to a manageable size
def resize_image(image, max_width=1024):
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image

# Improved restoration: white balance + CLAHE + gamma correction
def restore_underwater_image(image):
    try:
        result = cv2.xphoto.createSimpleWB().balanceWhite(image)
    except:
        result = image

    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    gamma = 1.2
    look_up = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    result = cv2.LUT(result, look_up)

    return result

# Check if uploaded file is a valid image
def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except:
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    if not file:
        return "No file uploaded", 400

    unique_name = str(uuid.uuid4()) + '_' + file.filename
    upload_path = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(upload_path)

    # Validate image
    if not is_valid_image(upload_path):
        os.remove(upload_path)
        return "Invalid image file", 400

    img = cv2.imread(upload_path)
    img = resize_image(img)
    restored_img = restore_underwater_image(img)

    restored_name = 'restored_' + unique_name
    restored_path = os.path.join(PROCESSED_FOLDER, restored_name)
    cv2.imwrite(restored_path, restored_img)

    response = render_template('result.html', original=upload_path, restored=restored_path)

    # Optional: cleanup after sending (better with scheduled task in production)
    # os.remove(upload_path)
    # os.remove(restored_path)

    return response

if __name__ == '__main__':
    app.run(debug=True)
