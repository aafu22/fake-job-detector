import pandas as pd


class DataValidation:

    def __init__(self):
        self.train_path = "artifacts/train.csv"

    def validate_data(self):

        df = pd.read_csv(self.train_path)

        print("=" * 50)
        print("DATA VALIDATION REPORT")
        print("=" * 50)

        print("\nDataset Shape:")
        print(df.shape)

        print("\nMissing Values:")
        print(df.isnull().sum())

        print("\nDuplicate Rows:")
        print(df.duplicated().sum())

        print("\nTarget Distribution:")
        print(df["fraudulent"].value_counts())

        print("\nValidation Completed Successfully")


if __name__ == "__main__":
    validator = DataValidation()
    validator.validate_data()