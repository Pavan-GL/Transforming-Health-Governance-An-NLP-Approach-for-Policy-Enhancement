import os
import logging
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Setup logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(log_dir, 'training.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def train_model(train_df):
    try:
        logging.info("Initializing the tokenizer and model.")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

        logging.info("Tokenizing training data.")
        train_encodings = tokenizer(list(train_df['cleaned_text']), truncation=True, padding=True)
        train_labels = train_df['label'].values.tolist()

        train_dataset = TextDataset(train_encodings, train_labels)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=16,
            save_steps=10_000,
            save_total_limit=2,
            logging_dir='./logs',
            logging_steps=500,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )

        logging.info("Starting training.")
        trainer.train()
        model.save_pretrained('D:/Transforming Health Governance An NLP Approach for Policy Enhancement/data/sentiment_model')
        logging.info("Model trained and saved successfully.")
        return model

    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

def evaluate_model(model, test_df):
    try:
        logging.info("Initializing the tokenizer for evaluation.")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        logging.info("Tokenizing test data.")
        test_encodings = tokenizer(list(test_df['cleaned_text']), truncation=True, padding=True)

        inputs = torch.tensor(test_encodings['input_ids'])
        logging.info("Making predictions.")
        with torch.no_grad():
            outputs = model(inputs)
        
        predictions = torch.argmax(outputs.logits, dim=-1)
        logging.info("Predictions made successfully.")
        return predictions.numpy()  # Return as numpy array for easier handling

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise

"""if __name__ == "__main__":
    try:
        logging.info("Loading training and testing data.")
        train_df = pd.read_csv('D:/Transforming Health Governance An NLP Approach for Policy Enhancement/data/train_data.csv')
        test_df = pd.read_csv('D:/Transforming Health Governance An NLP Approach for Policy Enhancement/data/test_data.csv')

        logging.info("Training the model.")
        model = train_model(train_df)

        logging.info("Evaluating the model.")
        predictions = evaluate_model(model, test_df)

        # Optionally save predictions as a CSV or pickle
        predictions_df = pd.DataFrame(predictions, columns=['predictions'])
        predictions_df.to_csv('D:/Transforming Health Governance An NLP Approach for Policy Enhancement/data/predictions.csv', index=False)
        logging.info("Predictions saved successfully.")

    except Exception as e:
        logging.error(f"Error in main execution: {e}")"""
