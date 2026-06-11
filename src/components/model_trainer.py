import os
import re
import joblib
import pandas as pd
import nltk
import mlflow

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split


try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


class ModelTrainer:

    def clean_text(self, text):

        text = str(text).lower()

        text = re.sub(r"http\S+", "", text)

        text = re.sub(r"[^a-zA-Z ]", " ", text)

        stop_words = set(stopwords.words("english"))

        words = [
            word
            for word in text.split()
            if word not in stop_words
        ]

        return " ".join(words)

    def train_model(self):

        df = pd.read_csv("artifacts/train.csv")

        text_columns = [
            "title",
            "company_profile",
            "description",
            "requirements",
            "benefits"
        ]

        df[text_columns] = df[text_columns].fillna("")

        df["combined_text"] = (
            df["title"] + " " +
            df["company_profile"] + " " +
            df["description"] + " " +
            df["requirements"] + " " +
            df["benefits"]
        )

        df["combined_text"] = df["combined_text"].apply(
            self.clean_text
        )

        X = df["combined_text"]
        y = df["fraudulent"]

        vectorizer = TfidfVectorizer(
            max_features=5000
        )

        X_vectorized = vectorizer.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized,
            y,
            test_size=0.2,
            random_state=42
        )

        model = LogisticRegression(
            class_weight="balanced",
            max_iter=1000
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        report = classification_report(
            y_test,
            predictions,
            output_dict=True
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        os.makedirs("models", exist_ok=True)

        joblib.dump(
            model,
            "models/model.pkl"
        )

        joblib.dump(
            vectorizer,
            "models/vectorizer.pkl"
        )

        mlflow.set_experiment(
            "Fake Job Detector"
        )

        with mlflow.start_run():

            mlflow.log_param(
                "model",
                "LogisticRegression"
            )

            mlflow.log_param(
                "max_features",
                5000
            )

            mlflow.log_param(
                "class_weight",
                "balanced"
            )

            mlflow.log_metric(
                "accuracy",
                accuracy
            )

            mlflow.log_metric(
                "fake_job_precision",
                report["1"]["precision"]
            )

            mlflow.log_metric(
                "fake_job_recall",
                report["1"]["recall"]
            )

            mlflow.log_metric(
                "fake_job_f1",
                report["1"]["f1-score"]
            )

            mlflow.log_artifact(
                "models/model.pkl"
            )

            mlflow.log_artifact(
                "models/vectorizer.pkl"
            )

        print("\nClassification Report:\n")

        print(
            classification_report(
                y_test,
                predictions
            )
        )

        print("\nModel Saved Successfully")
        print("MLflow Run Logged Successfully")


if __name__ == "__main__":

    trainer = ModelTrainer()

    trainer.train_model()