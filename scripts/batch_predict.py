import os
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification

def predict_batch(input_file, output_file, model_dir="model_output"):
    print(f"Loading model from {model_dir}...")
    try:
        tokenizer = BertTokenizer.from_pretrained(model_dir)
        model = BertForSequenceClassification.from_pretrained(model_dir)
    except OSError:
        print(f"Error: Could not find model in {model_dir}. Please train the model first.")
        return

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    print(f"Reading input from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file {input_file} not found.")
        return

    if 'text' not in df.columns:
        print("Error: Input CSV must have a 'text' column.")
        return

    predictions = []
    confidences = []

    print("Running predictions...")
    BATCH_SIZE = 16
    texts = df['text'].tolist()
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            
            # Assuming index 0 is negative, 1 is positive (standard for binary)
            # We need to confirm this mapping, usually 0=neg, 1=pos in older imdb scripts, 
            # but let's stick to simple mapping for now.
            # Label 0: Negative, Label 1: Positive
            
            vals, indices = torch.max(probs, dim=1)
            
            for val, idx in zip(vals, indices):
                sentiment = "positive" if idx.item() == 1 else "negative"
                predictions.append(sentiment)
                confidences.append(val.item())

    df['predicted_sentiment'] = predictions
    df['confidence'] = confidences

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Saving results to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch Sentiment Prediction')
    parser.add_argument('--input-file', required=True, help='Path to input CSV file')
    parser.add_argument('--output-file', required=True, help='Path to output CSV file')
    args = parser.parse_args()

    predict_batch(args.input_file, args.output_file)
