from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Model load karna
model = tf.keras.models.load_model("WheatDiseasesDetection.h5")

@app.route("/predict", methods=["POST"])
def predict():
    print("📢 Received a request!")  # Check karein request aayi ya nahi
    print("📢 Request files:", request.files)  # Check karein request me file hai ya nahi
    
    if "file" not in request.files:
        print("❌ No file found in request!")  
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    print("📢 File received:", file.filename)  # Check karein file sahi aa rahi hai ya nahi

    if file.filename == "":
        print("❌ No file selected!")  
        return jsonify({"error": "No file selected"}), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        print("✅ Image successfully opened!")  # Debugging ke liye print karein

        image = image.convert("RGB")  # PNG images me transparency hoti hai, isliye RGB me convert karein
        image = image.resize((255, 255))  # Model ke input size ke mutabiq resize karein
        img_array = np.array(image) / 255.0  # Normalize karein
        img_array = np.expand_dims(img_array, axis=0)  # Batch dimension add karein

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
    print("🚀 Starting Flask server...")  
    app.run(debug=True)

