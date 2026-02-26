import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EvalPrediction, PreTrainedTokenizerFast
import evaluate
from typing import cast, Dict, Any

def compute_metrics(eval_pred: EvalPrediction) -> dict:
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)
    res = metric.compute(predictions=predictions, references=labels)
    return cast(dict, res)

def main():
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    print("Loading custom dataset...")
    data_path = os.path.join(project_root, "data", "sentiment_training_data.csv")
    df = pd.read_csv(data_path)
    
    # Map labels to integers
    label_mapping = {"Negative": 0, "Neutral": 1, "Positive": 2, "Sarcastic": 3}
    df["label"] = df["label"].map(label_mapping)
    
    # Convert to HuggingFace Dataset
    dataset = Dataset.from_pandas(df)
    
    # Split into train/test (80/20) - for this tiny dataset it'll be very small
    dataset_dict = dataset.train_test_split(test_size=0.2)
    assert isinstance(dataset_dict, DatasetDict)
    
    model_name = "roberta-base"
    print(f"Loading foundation model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    assert tokenizer is not None
    
    def tokenize_function(examples: dict) -> dict:
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128) # type: ignore
    
    print("Tokenizing data...")
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)
    assert isinstance(tokenized_datasets, DatasetDict)
    
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]
    assert isinstance(train_dataset, Dataset)
    assert isinstance(eval_dataset, Dataset)
    
    # Initialize the model with 4 labels
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=4,
        id2label={v: k for k, v in label_mapping.items()},
        label2id=label_mapping
    )
    
    model_out_dir = os.path.join(project_root, "models", "custom-sentiment")
    os.makedirs(model_out_dir, exist_ok=True)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_out_dir,
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        # Reduce logging for this tiny sample run
        logging_steps=1,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("Starting fine-tuning process...")
    # This is the actual magic that trains the model on your data
    trainer.train()
    
    final_model_dir = os.path.join(model_out_dir, "final_model")
    print("Training complete! Saving your custom model...")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir) # type: ignore
    print(f"Saved to {final_model_dir}")

if __name__ == "__main__":
    main()
