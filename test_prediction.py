from src.pipeline.prediction_pipeline import PredictionPipeline

pipeline = PredictionPipeline()

job_text = """
Work from home.
Earn Rs 50000 per day.
No experience required.
Apply immediately.
"""

prediction, probability = pipeline.predict(job_text)

print("Prediction:", prediction)
print("Probability:", probability)