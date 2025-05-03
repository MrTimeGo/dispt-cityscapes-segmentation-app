from flask import Flask
from flask_cors import CORS

from .img_processing import img

app = Flask(__name__)
cors = CORS(app, supports_credentials=True, allow_headers="*")

app.register_blueprint(img)



