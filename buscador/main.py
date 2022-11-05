import os
import io
import secrets
import requests

from flask import request, redirect, url_for, render_template, send_from_directory
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import torch

from app import app, API_URL
from utils import allowed_file, get_closest_match


load_dotenv()

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER')
CATALOG_PATH = os.getenv('CATALOG_PATH')


@app.route('/')
def index_form():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def index_image():
    if 'file' not in request.files:
        error = 'No se envió ningún archivo'
        return render_template('index.html', error=error)
    file = request.files['file']
    if file.filename == '':
        error = 'No se seleccionó ningún archivo'
        return render_template('index.html', error=error)
    if file and allowed_file(file.filename):
        # hash para evitar sobreescribir
        filename = secrets.token_hex(nbytes=8) + '_' + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        files = {'file': open(filepath, 'rb')}
        apicall = requests.post(API_URL, files=files)
        if apicall.status_code == 200:
            error = None
            features = torch.load(io.BytesIO(apicall.content))
            closest_match = get_closest_match(features)
            result = {'closest_filename': closest_match}
        else:
            error = 'Error al procesar la imagen'
            result = None
        return render_template('index.html', filename=filename, result=result, error=error)
    else:
        error = 'Archivo no permitido. Solo se permite JPG, JPEG o PNG.'
        return render_template('index.html', error=error)


@app.route('/display/<filename>')
def display_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/catalogo/<filename>')
def display_match(filename):
    return send_from_directory(CATALOG_PATH, filename)


if __name__ == "__main__":
    app.run(port=5000)
