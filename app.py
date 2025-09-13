import os
import numpy as np
import tensorflow as tf
import mysql.connector
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image

# ✅ Initialize Flask App
app = Flask(__name__)

# ✅ MySQL Configuration
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "dental_db"
}

def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)

# ✅ Model Path
MODEL_PATH = "dental_classifier.tflite"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")

# ✅ Load Model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ✅ Upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), "temp")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Fetch doctor name
def get_doctor_name(doctor_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM doctorsignup WHERE doctorid = %s", (doctor_id,))
        name = cursor.fetchone()
        conn.close()
        return name[0] if name else "Unknown Doctor"
    except Exception as e:
        print(f"❌ Doctor lookup failed: {e}")
        return "Unknown Doctor"

# ✅ Fetch patient details
def get_patient_details(patient_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT patient_name, appointment_date FROM add_patient WHERE patientid = %s", (patient_id,))
        details = cursor.fetchone()
        conn.close()
        return details if details else ("Unknown Patient", "Unknown Date")
    except Exception as e:
        print(f"❌ Patient lookup failed: {e}")
        return ("Unknown Patient", "Unknown Date")

# ✅ Image preprocessing
def preprocess_image(image_path):
    try:
        image = Image.open(image_path).convert("L")
        expected_height = input_details[0]['shape'][1]
        expected_width = input_details[0]['shape'][2]
        image = image.resize((expected_width, expected_height))
        image = np.array(image).astype(np.float32) / 255.0
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        print(f"❌ Preprocessing error: {e}")
        return None

@app.route("/test", methods=["GET"])
def test():
    return jsonify({
        "message": "✅ Server is running",
        "input_shape": input_details[0]['shape'].tolist()  # convert ndarray -> list
    }), 200

# ✅ Route: /get_model
@app.route("/get_model", methods=["GET"])
def get_model():
    try:
        return send_file(MODEL_PATH, as_attachment=True)
    except Exception as e:
        return jsonify({"error": f"Model send failed: {str(e)}"}), 500

# ✅ Route: /predict
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Image not found"}), 400

        file = request.files["image"]
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        doctor_id = request.form.get("doctorid")
        patient_id = request.form.get("patientid")
        if not doctor_id or not patient_id:
            return jsonify({"error": "doctorid and patientid are required"}), 400

        doctor_name = get_doctor_name(doctor_id)
        patient_name, appointment_date = get_patient_details(patient_id)

        image_data = preprocess_image(file_path)
        if image_data is None:
            return jsonify({"error": "Image preprocessing failed"}), 500

        interpreter.set_tensor(input_details[0]["index"], image_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]["index"])[0][0]

        classification = "Thick" if prediction * 10 > 2.0 else "Thin"

        return jsonify({
            "doctor_name": doctor_name,
            "patient_name": patient_name,
            "appointment_date": appointment_date,
            "prediction": round(float(prediction), 4),
            "classification": classification
        }), 200

    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

# ✅ Start server
if __name__ == "__main__":
    print("🚀 Server running with model:", MODEL_PATH)
    app.run(host="0.0.0.0", port=5000, debug=True)
