import re
import joblib
import pandas as pd
import nltk

from nltk.corpus import stopwords
from sklearn.metrics import classification_report

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


class ModelEvaluation:

    def clean_text(self, text):

        text = str(text).lower()

        text = re.sub(r"http\\S+", "", text)

        text = re.sub(r"[^a-zA-Z ]", " ", text)

        stop_words = set(stopwords.words("english"))

        words = [
            word
            for word in text.split()
            if word not in stop_words
        ]

        return " ".join(words)

    def evaluate_model(self):

        df = pd.read_csv("artifacts/test.csv")

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

        vectorizer = joblib.load(
            "models/vectorizer.pkl"
        )

        model = joblib.load(
            "models/model.pkl"
        )

        X_vectorized = vectorizer.transform(X)

        predictions = model.predict(X_vectorized)

        print("\nMODEL EVALUATION REPORT\n")

        print(classification_report(y, predictions))


if __name__ == "__main__":
    evaluator = ModelEvaluation()
    evaluator.evaluate_model()