import os
import argparse
import json
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset

# Set up arguments
parser = argparse.ArgumentParser(description='Train BERT model')
parser.add_argument('--debug', action='store_true', help='Run in debug mode with a small dataset subset')
args = parser.parse_args()

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    print(f"Starting training (Debug mode: {args.debug})")

    # Paths
    data_dir = os.path.join("data", "processed")
    model_output_dir = "model_output"
    results_dir = "results"
    
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    print("Loading data...")
    try:
        df_train = pd.read_csv(os.path.join(data_dir, "train.csv"))
        df_test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    except FileNotFoundError:
        print("Data files not found. Please run scripts/preprocess.py first.")
        return

    if args.debug:
        print("DEBUG MODE: Using only 100 samples.")
        df_train = df_train.head(100)
        df_test = df_test.head(20)

    # Convert to HF Dataset
    train_dataset = Dataset.from_pandas(df_train)
    test_dataset = Dataset.from_pandas(df_test)

    # Tokenizer
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding=True, max_length=128)

    print("Tokenizing data...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # We need to set the format for pytorch
    train_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Model
    print("Loading model...")
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir='./results_checkpoints',
        learning_rate=2e-5,
        per_device_train_batch_size=8 if not args.debug else 2,
        per_device_eval_batch_size=16 if not args.debug else 2,
        num_train_epochs=3 if not args.debug else 1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no", # We save explicitly at the end
        use_cpu=True if args.debug else not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Training...")
    trainer.train()

    # Evaluate
    print("Evaluating...")
    eval_results = trainer.evaluate()
    
    # Save metrics
    metrics = {
        "accuracy": eval_results["eval_accuracy"],
        "precision": eval_results["eval_precision"],
        "recall": eval_results["eval_recall"],
        "f1_score": eval_results["eval_f1_score"]
    }
    
    metrics_file = os.path.join(results_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
        
    # Save Run Summary (Experiment Tracking)
    run_summary = {
        "hyperparameters": {
            "model_name": model_name,
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "num_epochs": training_args.num_train_epochs
        },
        "final_metrics": metrics
    }
    summary_file = os.path.join(results_dir, "run_summary.json")
    with open(summary_file, "w") as f:
        json.dump(run_summary, f, indent=4)

    print(f"metrics saved to {metrics_file}")
    print(f"summary saved to {summary_file}")

    # Save Model Artifacts
    print(f"Saving model artifacts to {model_output_dir}...")
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)
    print("Done.")

if __name__ == "__main__":
    main()
