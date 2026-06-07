import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path):
    return pd.read_csv(file_path)


def handle_pseudo_missing_values(df):
    columns_with_zero = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    for col in columns_with_zero:
        df[col] = df[col].replace(0, df[col].median())

    return df


def split_data(df, target_column="Outcome"):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train_df = X_train.copy()
    train_df[target_column] = y_train

    test_df = X_test.copy()
    test_df[target_column] = y_test

    return train_df, test_df


def save_data(train_df, test_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)


def preprocess_data(input_path, output_dir):
    df = load_data(input_path)
    df = handle_pseudo_missing_values(df)
    train_df, test_df = split_data(df)
    save_data(train_df, test_df, output_dir)

    print("Preprocessing completed successfully.")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")


if __name__ == "__main__":
    input_path = "../dataset_raw/diabetes.csv"
    output_dir = "diabetes_preprocessing"

    preprocess_data(input_path, output_dir)