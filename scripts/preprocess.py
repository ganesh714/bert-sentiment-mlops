import os
import pandas as pd
from datasets import load_dataset
import shutil

def preprocess_data():
    print("Loading IMDB dataset...")
    # Load dataset from Hugging Face
    dataset = load_dataset("imdb")

    # Access train and test splits
    train_data = dataset["train"]
    test_data = dataset["test"]

    print(f"Train dataset size: {len(train_data)}")
    print(f"Test dataset size: {len(test_data)}")

    # Convert to pandas DataFrame for easier CSV saving
    # The dataset has 'text' and 'label' columns
    df_train = pd.DataFrame(train_data)
    df_test = pd.DataFrame(test_data)

    # Basic cleaning (optional, as IMDB from HF is relatively clean, but let's remove some br tags)
    print("Cleaning text...")
    df_train['text'] = df_train['text'].str.replace('<br />', ' ', regex=False)
    df_test['text'] = df_test['text'].str.replace('<br />', ' ', regex=False)

    # Create processed directory if not exists
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    print(f"Saving processed data to {output_dir}...")
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print("Preprocessing complete.")

if __name__ == "__main__":
    preprocess_data()
