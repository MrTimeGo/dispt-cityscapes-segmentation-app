import io
import numpy as np

from flask import Blueprint, request, jsonify, send_file
from PIL import Image

from .model.evaluate import evaluate

img = Blueprint('images', __name__)

@img.route("/images", methods=['POST'])
def process_image():
    print("Files in request:", request.files)
    print("Content-Type:", request.headers.get('Content-Type'))
    
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']

    try:
        image = Image.open(image_file)
        width, height = image.size
    except Exception as e:
        return jsonify({"error": "Invalid image file", "details": str(e)}), 400

    print("Received image file:", image_file.filename)

    SIZE = 160

    skip_labels = request.args.getlist('skip_labels')

    if width < SIZE or height < SIZE:
        return jsonify({"error": "Invalid image size. Width and height should be more than 160"}), 400

    output = evaluate(np.array(image), skip_labels)
    return send_file(
        io.BytesIO(output),
        mimetype='image/png',
        as_attachment=True,
        download_name=f"{image_file.name}_segmented.png",
    )