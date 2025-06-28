from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io, os

# Paths
STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')
MODEL_PATH = os.getenv('MODEL_PATH', os.path.join(STATIC_DIR, 'anemia_model_best.h5'))

# Initialize app
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path='')
CORS(app)
model = load_model(MODEL_PATH)

IMG_SIZE = (224, 224)  # adapt to your model

def preprocess_image(img_bytes):
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        img_bytes = file.read()
        inp = preprocess_image(img_bytes)
        preds = model.predict(inp)[0]
        idx = int(np.argmax(preds))
        conf = float(np.max(preds)*100)
        label = 'Non-Anemic' if idx == 0 else 'Anemic'
        return jsonify({'prediction': label, 'confidence': round(conf, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Serve spa/static
@app.route('/')
def root():
    return send_from_directory(app.static_folder, 'index.html')

@app.errorhandler(404)
def not_found(e):
    # Attempt to serve any file from static, else index.html for SPA routes
    path = request.path.lstrip('/')
    file_path = os.path.join(app.static_folder, path)
    if os.path.exists(file_path):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
