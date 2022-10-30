from app import app
from utils import get_prediction, get_features
from flask import Flask, jsonify, request
import base64
import io
import torch


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()

        # Obtenemos las features
        deep_features = get_features(image_bytes=img_bytes)
        # print(deep_features)

        # Las pasamos a bytes usando un buffer I/O
        buff = io.BytesIO()
        torch.save(deep_features, buff)
        buff.seek(0)
        return buff  # return buff.read()


if __name__ == '__main__':
    app.run(port=5001)
