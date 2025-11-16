import kagglehub
import pandas as pd

from sklearn.preprocessing import LabelEncoder

def load_heart_failure_dataset(path, target_col):
    """Downloads and returns the Heart Failure dataset as a Pandas DataFrame."""

    dataset_id = path

    path = kagglehub.dataset_download(dataset_id)

    csv_path = f"{path}/heart.csv"
    df = pd.read_csv(csv_path)

    # separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # encode categorical columns
    for col in X.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    return df, X, y


if __name__ == "__main__":
    df = load_heart_failure_dataset()
    print("✅ Dataset loaded successfully")