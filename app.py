from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import gdown  # ✅ Required to download from Google Drive

app = Flask(__name__)

# Model file name and Google Drive file ID
MODEL_FILE = "WheatDiseasesDetection.h5"
DRIVE_FILE_ID = "1uvabWaPKnBuX7ND3POH6-ZlgOnufdSxJ"

# 🔽 Model download if not exists
if not os.path.exists(MODEL_FILE):
    print("📦 Model not found. Downloading from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_FILE, quiet=False)
    print("✅ Model downloaded.")

# ✅ Load model
print("📥 Loading model...")
model = tf.keras.models.load_model(MODEL_FILE)
print("✅ Model loaded.")

@app.route("/predict", methods=["POST"])
def predict():
    print("📢 Received a request!")
    print("📢 Request files:", request.files)
    
    if "file" not in request.files:
        print("❌ No file found in request!")  
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    print("📢 File received:", file.filename)

    if file.filename == "":
        print("❌ No file selected!")  
        return jsonify({"error": "No file selected"}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        print("✅ Image successfully opened!")

        image = image.convert("RGB")
        image = image.resize((255, 255))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        pred_index = np.argmax(prediction, axis=1)[0]

        conditions = {
            0: 'Aphid', 1: 'Black Rust', 2: 'Blast', 3: 'Brown Rust',
            4: 'Common Root Rot', 5: 'Fusarium Head Blight', 6: 'Healthy',
            7: 'Leaf Blight', 8: 'Mildew', 9: 'Mite', 10: 'Septoria',
            11: 'Smut', 12: 'Stem Fly', 13: 'Tan Spot', 14: 'Yellow Rust'
        }

        print("✅ Prediction successful:", conditions.get(pred_index, "Unknown Disease"))
        return jsonify({"prediction": conditions.get(pred_index, "Unknown Disease")})
    
    except Exception as e:
        print("❌ Error processing image:", str(e))
        return jsonify({"error": "Failed to process image"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting Flask server on port {port}...")
    app.run(host="0.0.0.0", port=port)


        
           
