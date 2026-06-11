import os
import pandas as pd

from sklearn.model_selection import train_test_split


class DataIngestion:

    def __init__(self):
        self.data_path = "data/raw/fake_job_postings.csv"
        self.artifacts_path = "artifacts"

    def initiate_data_ingestion(self):

        print("Reading dataset...")

        df = pd.read_csv(self.data_path)

        os.makedirs(self.artifacts_path, exist_ok=True)

        train_set, test_set = train_test_split(
            df,
            test_size=0.2,
            random_state=42
        )

        train_path = os.path.join(self.artifacts_path, "train.csv")
        test_path = os.path.join(self.artifacts_path, "test.csv")

        train_set.to_csv(train_path, index=False)
        test_set.to_csv(test_path, index=False)

        print("Train file saved:", train_path)
        print("Test file saved:", test_path)

        print(f"Train Shape: {train_set.shape}")
        print(f"Test Shape: {test_set.shape}")


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()