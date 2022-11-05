import os

from flask import Flask
from dotenv import load_dotenv

load_dotenv()

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER')

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024
API_URL = 'http://127.0.0.1:5001/predict'