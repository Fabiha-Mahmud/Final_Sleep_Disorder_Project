from flask import Flask, render_template, request, send_file
import numpy as np
import pandas as pd
import joblib
from fpdf import FPDF
import os

# -------------------------------------------------
# App Initialization
# -------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------
# Load Model + Metrics (DYNAMIC)
# -------------------------------------------------
bundle = joblib.load("model_bundle.pkl")

model = bundle["model"]
metrics = bundle["metrics"]

# -------------------------------------------------
# Label Mapping
# -------------------------------------------------
LABEL_MAP = {
    0: "No Sleep Disorder",
    1: "Insomnia",
    2: "Sleep Apnea"
}

# -------------------------------------------------
# Home Page
# -------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------------------------------
# Prediction Route
# -------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect user inputs (same order as training)
        input_data = [
            int(request.form["gender"]),
            int(request.form["age"]),
            int(request.form["occupation"]),
            float(request.form["sleep_duration"]),
            int(request.form["quality"]),
            int(request.form["stress"]),
            int(request.form["bmi"]),
            int(request.form["heart"]),
            int(request.form["sys"]),
            int(request.form["dia"])
        ]

        # Convert to NumPy array
        input_array = np.array(input_data).reshape(1, -1)

        # Prediction
        prediction = model.predict(input_array)[0]
        probability = np.max(model.predict_proba(input_array)) * 100

        # Render report page
        return render_template(
            "report.html",
            prediction=LABEL_MAP[prediction],
            confidence=round(probability, 2),

            accuracy=round(metrics["accuracy"] * 100, 2),
            precision=round(metrics["precision"] * 100, 2),
            recall=round(metrics["recall"] * 100, 2),
            f1=round(metrics["f1"] * 100, 2)
        )

    except Exception as e:
        return f"Prediction Error: {e}"

# -------------------------------------------------
# Download Report
# -------------------------------------------------
@app.route("/download")
def download():
    # Create FPDF object
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)

    # Title
    pdf.cell(0, 10, "Sleep Disorder Prediction Report", ln=True, align="C")
    pdf.ln(10)

    # Metrics
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 8, f"Accuracy : {round(metrics['accuracy'] * 100, 2)}%", ln=True)
    pdf.cell(0, 8, f"Precision: {round(metrics['precision'] * 100, 2)}%", ln=True)
    pdf.cell(0, 8, f"Recall   : {round(metrics['recall'] * 100, 2)}%", ln=True)
    pdf.cell(0, 8, f"F1 Score : {round(metrics['f1'] * 100, 2)}%", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 6, "This report is generated using the trained Voting Ensemble Model for Sleep Disorder Prediction. Metrics are dynamically loaded from offline evaluation.")

    # Save PDF
    file_path = "sleep_disorder_report.pdf"
    pdf.output(file_path)

    # Send PDF as download
    return send_file(file_path, as_attachment=True)

# -------------------------------------------------
# Run App
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
