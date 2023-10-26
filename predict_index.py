import torch
from transformers import MPNetForSequenceClassification, MPNetConfig, MPNetTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from transformers import TrainingArguments, Trainer

def load_data(file_path):
    """
    Load the CSV data using pandas
    """
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data, num_cols):
    """
    Preprocess the data
    Normalize the numerical columns
    """
    scaler = StandardScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])
    return data

def split_data(data, test_size):
    """
    Split the data into training and validation sets
    """
    train_data, val_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, val_data

def train_model(train_encodings, train_labels, val_encodings, val_labels, model, training_args):
    """
    Train the model
    """
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']),
                                                   torch.tensor(train_encodings['attention_mask']),
                                                   train_labels)
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_encodings['input_ids']),
                                                 torch.tensor(val_encodings['attention_mask']),
                                                 val_labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    return trainer

def evaluate_model(trainer):
    """
    Evaluate the model
    """
    eval_results = trainer.evaluate()
    return eval_results

def save_model(model, model_path):
    """
    Save the trained model
    """
    model.save_pretrained(model_path)

def load_model(model_path):
    """
    Load the saved model
    """
    model = MPNetForSequenceClassification.from_pretrained(model_path)
    return model

def preprocess_input_data(input_data, num_cols, scaler):
    """
    Preprocess the input data for inference
    """
    input_data[num_cols] = scaler.transform(input_data[num_cols])
    return input_data

def predict_stock_price(model, input_encodings):
    """
    Use the model to predict the next day's stock price
    """
    input_ids = torch.tensor(input_encodings['input_ids'])
    attention_mask = torch.tensor(input_encodings['attention_mask'])
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    predictions = outputs.logits
    return predictions

def main():
    # Load the CSV data using pandas
    data = load_data('data.csv')

    # Preprocess the data
    num_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    data = preprocess_data(data, num_cols)

    # Split the data into training and validation sets
    train_data, val_data = split_data(data, test_size=0.2)

    # Define the model architecture
    config = MPNetConfig.from_pretrained('microsoft/mps-roberta-large')
    config.num_labels = 1
    model = MPNetForSequenceClassification(config)

    # Train the model
    tokenizer = MPNetTokenizer.from_pretrained('microsoft/mps-roberta-large')
    train_encodings = tokenizer(list(train_data[num_cols].T.to_dict().values()), truncation=True, padding=True)
    val_encodings = tokenizer(list(val_data[num_cols].T.to_dict().values()), truncation=True, padding=True)

    train_labels = torch.tensor(train_data['Close'].values)
    val_labels = torch.tensor(val_data['Close'].values)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        evaluation_strategy='steps',
        eval_steps=500,
        metric_for_best_model='mse',
        greater_is_better=False
    )

    trainer = train_model(train_encodings, train_labels, val_encodings, val_labels, model, training_args)

    # Evaluate the model
    eval_results = evaluate_model(trainer)

    # Save the trained model
    save_model(model, 'model')

    # Load the saved model
    model = load_model('model')

    # Preprocess the input data for inference
    input_data = pd.read_csv('input.csv')
    scaler = StandardScaler()
    scaler.fit(data[num_cols])
    input_data = preprocess_input_data(input_data, num_cols, scaler)
    input_encodings = tokenizer(list(input_data[num_cols].T.to_dict().values()), truncation=True, padding=True)

    # Use the model to predict the next day's stock price
    predictions = predict_stock_price(model, input_encodings)
    print(predictions)

if __name__ == '__main__':
    main()
