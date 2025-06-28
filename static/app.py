from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io, os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (update path if you move the file)
MODEL_PATH = os.getenv("MODEL_PATH", "anemia_model_best.h5")
model = load_model(MODEL_PATH)

# Image preprocessing
IMG_SIZE = (224, 224)  # Adjust if your model expects a different input size

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0  # Normalization
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        img_bytes = file.read()
        input_tensor = preprocess_image(img_bytes)
        preds = model.predict(input_tensor)[0]

        # Modify this logic if your model's output is different
        class_idx = int(np.argmax(preds))
        confidence = float(np.max(preds) * 100)
        label = "Non-Anemic" if class_idx == 0 else "Anemic"

        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
