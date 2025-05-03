from flask import Blueprint, request, jsonify
from PIL import Image

img = Blueprint('images', __name__)

@img.route("/images", methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']

    try:
        image = Image.open(image_file)
        width, height = image.size
    except Exception as e:
        return jsonify({"error": "Invalid image file", "details": str(e)}), 400

    print("Received image file:", image_file.filename)

    # TODO: invoke the segmentation model here

    mock_response = {
        "imgHeight": width,
        "imgWidth": height,
        "objects": [
            {
                "label": "road",
                "polygon": [
                    [0, 769],
                    [1, 290],
                    [574, 93],
                    [528, 1],
                    [0, 524],
                    [1, 0],
                    [448, 0]
                ]
            }
        ]
    }

    return jsonify(mock_response)