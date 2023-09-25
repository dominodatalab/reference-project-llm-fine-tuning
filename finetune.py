import sys
import torch
import transformers
import json
import argparse

import numpy as np
import pandas as pd 

from transformers import BertTokenizer, Trainer, BertForSequenceClassification, TrainingArguments, pipeline

from datasets import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def load_data(file_name = "all-data.csv"):
    # Load from CSV
    df = pd.read_csv(file_name, delimiter=",", encoding="latin-1", header=None).fillna("")
    df = df.rename(columns=lambda x: ["label", "sentence"][x])
    
    # Encode labels
    df["label"] = df["label"].replace(["neutral","positive","negative"],[0,1,2]) 
    return df

def split(df):
    df_train, df_test, = train_test_split(df, stratify=df["label"], test_size=0.1, random_state=42)
    df_train, df_val = train_test_split(df_train, stratify=df_train["label"],test_size=0.1, random_state=42)
    print("Samples in train      : {:d}".format(df_train.shape[0]))
    print("Samples in validation : {:d}".format(df_val.shape[0]))
    print("Samples in test       : {:d}".format(df_test.shape[0]))
    
    return df_train, df_val, df_test


def prep_datasets(df_train, df_val, df_test, model, tokenizer):
    dataset_train = Dataset.from_pandas(df_train)
    dataset_val = Dataset.from_pandas(df_val)
    dataset_test = Dataset.from_pandas(df_test)

    dataset_train = dataset_train.map(lambda e: tokenizer(e["sentence"], truncation=True, padding="max_length", max_length=315), batched=True)
    dataset_val = dataset_val.map(lambda e: tokenizer(e["sentence"], truncation=True, padding="max_length", max_length=315), batched=True)
    dataset_test = dataset_test.map(lambda e: tokenizer(e["sentence"], truncation=True, padding="max_length" , max_length=315), batched=True)

    dataset_train.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
    dataset_val.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
    dataset_test.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
    
    return dataset_train, dataset_val, dataset_test


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy" : accuracy_score(predictions, labels)}


def train(model, args, dataset_train, dataset_val):
    trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset_train,
            eval_dataset=dataset_val,
            compute_metrics=compute_metrics)
    
    trainer.train()
    
    return trainer

def main():
    
    parser = argparse.ArgumentParser(description="Fine-tuning a FinBERT model using the Sentiment Analysis for Financial News dataset. \
                                     This work is licensed \
                                     under the Creative Commons Attribution \
                                     4.0 International License.")
    
    parser.add_argument("--lr", help="Learning rate.", required=False, default=0.00001, type=float)
    parser.add_argument("--epochs", help="Training epochs.", required=False, default=1, type=int)
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        print("GPU acceleration is available!")
    else:
        print("GPU acceleration is NOT available! Training, fine-tuning, and inference speed will be adversely impacted.")
        
    df = load_data()
    df_train, df_val, df_test = split(df)
    
    model = BertForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone",num_labels=3)
    tokenizer = BertTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    
    dataset_train, dataset_val, dataset_test = prep_datasets(df_train, df_val, df_test, model, tokenizer)
    
    args = TrainingArguments(
            output_dir = "temp/",
            evaluation_strategy = "epoch",
            learning_rate=args.lr,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=args.epochs,
            weight_decay=0.01,
            metric_for_best_model="accuracy",
            save_total_limit = 2,
            save_strategy = "no",
            load_best_model_at_end=False,
            report_to = "none",
            optim="adamw_torch")

    trainer = train(model, args, dataset_train, dataset_val)
    
    accuracy_test = trainer.predict(dataset_test).metrics["test_accuracy"]
    print("Accuracy on test: {:.2f}".format(accuracy_test))

    # Please change the location to where you want to save the model, /mnt/artifacts is available for git based projects
    trainer.save_model("/mnt/artifacts/finbert-sentiment/")

    
if __name__ == "__main__":
    main()
