import pandas as pd
import re
import nltk

from nltk.corpus import stopwords

nltk.download("stopwords")


class DataTransformation:

    def __init__(self):
        self.train_path = "artifacts/train.csv"

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

    def transform_data(self):

        df = pd.read_csv(self.train_path)

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

        print(df["combined_text"].head())

        return df


if __name__ == "__main__":
    transformer = DataTransformation()
    transformer.transform_data()