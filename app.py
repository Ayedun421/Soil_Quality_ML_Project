from flask import Flask, render_template, request
import joblib
import numpy as np
import os


app = Flask(__name__)

# ---------------- LOAD MODEL FILES ----------------
MODEL_PATH = "model/soil_model.pkl"
SCALER_PATH = "model/scaler.pkl"
ENCODER_PATH = "model/label_encoder.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

# ---------------- ROUTES ----------------


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            # Debug: see what Flask receives
            print("FORM DATA:", request.form)

            sample = np.array([[
                float(request.form.get("ph", 0)),
                float(request.form.get("temp", 0)),
                float(request.form.get("moisture", 0)),
                float(request.form.get("humidity", 0)),
                float(request.form.get("nitrogen", 0)),
                float(request.form.get("phosphorus", 0)),
                float(request.form.get("potassium", 0)),
                float(request.form.get("rainfall", 0))
            ]])

            sample_scaled = scaler.transform(sample)
            pred = model.predict(sample_scaled)
            prediction = encoder.inverse_transform(pred)[0]

        except Exception as e:
            print("PREDICTION ERROR:", e)
            error = "Invalid input values. Please check your entries."

    return render_template(
        "predict.html",
        prediction=prediction,
        error=error
    )




if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


@app.route("/")
def home():
    return render_template("home.html")

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/dataset")
def dataset():
    return render_template("dataset.html")

@app.route("/settings")
def settings():
    return render_template("settings.html")

# ---------------- RUN APP ----------------
if __name__ == "__main__":
    app.run(debug=True)
