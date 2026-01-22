from flask import Flask, render_template, request
import os
import joblib
import pickle
import numpy as np

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'house_price_model.pkl')
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        features = [
            float(request.form["OverallQual"]),
            float(request.form["GrLivArea"]),
            float(request.form["TotalBsmtSF"]),
            float(request.form["GarageCars"]),
            float(request.form["FullBath"]),
            float(request.form["YearBuilt"]),
        ]

        prediction = model.predict([features])[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

