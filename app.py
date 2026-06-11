from flask import Flask, render_template, request

from src.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)


@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction_text=None,
        probability=None,
        risk_level=None,
        job_text=""
    )


@app.route("/predict", methods=["POST"])
def predict():

    job_text = request.form.get("job_text", "").strip()

    if not job_text:
        return render_template(
            "index.html",
            prediction_text=None,
            probability=None,
            risk_level=None,
            job_text=""
        )

    pipeline = PredictionPipeline()

    prediction, probability = pipeline.predict(job_text)

    probability_percent = round(probability * 100, 2)

    if prediction == 1:

        result = "Fake Job Posting"

        if probability >= 0.85:
            risk_level = "🔴 High Risk"
        elif probability >= 0.70:
            risk_level = "🟠 Medium Risk"
        else:
            risk_level = "🟡 Low Risk"

    else:

        result = "Real Job Posting"
        risk_level = "🟢 Low Risk"

    return render_template(
        "index.html",
        prediction_text=result,
        probability=probability_percent,
        risk_level=risk_level,
        job_text=job_text
    )


if __name__ == "__main__":
    app.run(debug=True)