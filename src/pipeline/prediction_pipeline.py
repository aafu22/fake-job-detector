import re
import joblib
import nltk

from nltk.corpus import stopwords

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


class PredictionPipeline:

    def __init__(self):

        self.model = joblib.load(
            "models/model.pkl"
        )

        self.vectorizer = joblib.load(
            "models/vectorizer.pkl"
        )

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

    def predict(self, text):

        cleaned_text = self.clean_text(text)

        transformed_text = self.vectorizer.transform(
            [cleaned_text]
        )

        prediction = self.model.predict(
            transformed_text
        )[0]

        probability = self.model.predict_proba(
            transformed_text
        ).max()

        return prediction, probability