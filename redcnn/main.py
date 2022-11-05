import io

from flask import request, send_file
import torch

from app import app
from utils import get_features


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()

        # Obtenemos las features
        deep_features = get_features(image_bytes=img_bytes)
        buffer = io.BytesIO()
        torch.save(deep_features, buffer)
        buffer.seek(0)
        return send_file(buffer, download_name="tensor_buffer")


if __name__ == '__main__':
    app.run(port=5001)
