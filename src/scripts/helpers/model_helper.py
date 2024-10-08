from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

class MetaDescriptionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: tensor[idx] for key, tensor in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
    

def encode_labels(data, column_name):
    """
    Encode the categorical labels into numerical labels using LabelEncoder.

    Args:
    - data (pd.DataFrame): The dataset containing the categorical column.
    - column_name (str): The name of the column to encode.

    Returns:
    - y (np.array): The encoded labels.
    - label_mapping (dict): A dictionary mapping the actual labels to encoded labels.
    - label_encoder (LabelEncoder): The fitted LabelEncoder instance for decoding.
    """
    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Fit the LabelEncoder and transform the specified column into numerical labels
    y = label_encoder.fit_transform(data[column_name])

    # Get the mapping of labels to their encoded values
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    # Print the mapping
    print("Mapping of actual labels to encoded labels:")
    for label, encoded_label in label_mapping.items():
        print(f"'{label}': {encoded_label}")
    
    return y, label_mapping, label_encoder

def decode_labels(predicted_labels, label_encoder):
    """
    Decode numerical labels back to their original categorical labels.

    Args:
    - predicted_labels (np.array): The predicted numerical labels.
    - label_encoder (LabelEncoder): The fitted LabelEncoder instance.

    Returns:
    - decoded_labels (np.array): The decoded original labels.
    """
    return label_encoder.inverse_transform(predicted_labels)

def stratified_split(data, target, feature_column, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
    """
    Perform a stratified split of the data into training, validation, and test sets.

    Args:
    - data (pd.DataFrame): The dataset containing the features.
    - target (np.array): The target labels.
    - feature_column (str): The name of the column to use as features.
    - train_size (float): Proportion of the data to include in the training set (default is 0.7).
    - val_size (float): Proportion of the data to include in the validation set (default is 0.15).
    - test_size (float): Proportion of the data to include in the test set (default is 0.15).
    - random_state (int): Seed used by the random number generator (default is 42).

    Returns:
    - X_train, X_val, X_test: Features for the training, validation, and test sets.
    - y_train, y_val, y_test: Labels for the training, validation, and test sets.
    """
    # Step 1: Define features
    X = data[feature_column]

    # Step 2: Perform Stratified Split
    # First split: train and temp (70% train, 30% temp)
    X_train, X_temp, y_train, y_temp = train_test_split(X, target, test_size=(1 - train_size), stratify=target, random_state=random_state)

    # Second split: Split the temp set into validation and test (each 15% of original data)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=random_state)

    # Step 3: Verify the splits
    print(f'Training set: {len(X_train)} samples')
    print(f'Validation set: {len(X_val)} samples')
    print(f'Test set: {len(X_test)} samples')

    return X_train, X_val, X_test, y_train, y_val, y_test

def tokenize_and_convert_labels(X_train, X_val, X_test, y_train, y_val, y_test, model_name='bert-base-uncased', max_length=64):
    """
    Tokenize the text data and convert labels to tensor format.

    Args:
    - X_train, X_val, X_test: Features for training, validation, and test sets.
    - y_train, y_val, y_test: Labels for training, validation, and test sets.
    - model_name (str): Name of the pre-trained BERT model to load the tokenizer (default is 'bert-base-uncased').
    - max_length (int): Maximum length for the tokenized sequences (default is 64).

    Returns:
    - train_encodings, val_encodings, test_encodings: Tokenized features for training, validation, and test sets.
    - train_labels, val_labels, test_labels: Labels converted to tensor format.
    """
    # Load the pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Tokenize the data
    def tokenize_texts(texts):
        return tokenizer(
            texts.tolist(),  # Convert Pandas Series to list
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

    # Tokenize the datasets
    train_encodings = tokenize_texts(X_train)
    val_encodings = tokenize_texts(X_val)
    test_encodings = tokenize_texts(X_test)

    # Convert labels to tensor format
    train_labels = torch.tensor(y_train.tolist())
    val_labels = torch.tensor(y_val.tolist())
    test_labels = torch.tensor(y_test.tolist())

    return train_encodings, val_encodings, test_encodings, train_labels, val_labels, test_labels
    
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    accuracy = accuracy_score(p.label_ids, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(p.label_ids, preds, average='weighted')
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

def initialize_model(model_name='bert-base-uncased', num_labels=2):
    """
    Initialize a pre-trained BERT model for sequence classification.

    Args:
    - model_name (str): The name of the pre-trained model (default is 'bert-base-uncased').
    - num_labels (int): The number of labels/classes in the classification task.

    Returns:
    - model (BertForSequenceClassification): The initialized BERT model.
    """
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return model

def evaluate_model_performance(labels, predicted_labels):
    """
    Calculate and print evaluation metrics for a classification model.

    Args:
    - labels (array-like): True labels.
    - predicted_labels (array-like): Predicted labels.

    Returns:
    - metrics (dict): A dictionary containing the calculated metrics.
    """
    # Calculate accuracy
    accuracy = accuracy_score(labels, predicted_labels)

    # Calculate precision
    precision = precision_score(labels, predicted_labels, average='weighted')

    # Calculate recall
    recall = recall_score(labels, predicted_labels, average='weighted')

    # Calculate F1 score
    f1 = f1_score(labels, predicted_labels, average='weighted')

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(labels, predicted_labels)

    # Print the calculated metrics
    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Confusion Matrix:\n{conf_matrix}')

    # Return the metrics in a dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

    return metrics

def evaluate_results(trainer, eval_dataset):
    """
    Evaluate the model, save the results to a CSV file, and return the evaluation DataFrame.

    Args:
    - trainer (Trainer): The Trainer instance to evaluate the model.
    - eval_dataset (Dataset): The dataset to evaluate on.
    - path_to_save_results (str): The path to save the evaluation results.
    - filename (str): The filename to save the results as (default is 'eval_results.csv').

    Returns:
    - eval_df (pd.DataFrame): The evaluation results as a DataFrame.
    """
    # Evaluate the model on the test dataset
    eval_results = trainer.evaluate(eval_dataset)

    # Convert the evaluation results to a DataFrame
    eval_df = pd.DataFrame([eval_results])

    return eval_df

def save_metrics_to_csv(metrics, average_metrics_file, class_metrics_file):
    """
    Save the average metrics and per-class metrics to separate CSV files.

    Args:
    - metrics (dict): The dictionary containing the calculated metrics.
    - average_metrics_file (str): The filename for the average metrics CSV.
    - class_metrics_file (str): The filename for the class-wise metrics CSV.
    """
    # Extract the average metrics
    average_metrics = {
        'accuracy': [metrics['accuracy']],
        'precision': [metrics['precision']],
        'recall': [metrics['recall']],
        'f1': [metrics['f1']]
    }

    # Convert the average metrics to a DataFrame and save to CSV
    average_metrics_df = pd.DataFrame(average_metrics)
    average_metrics_df.to_csv(average_metrics_file, index=False)
    print(f'Average metrics saved to {average_metrics_file}')

    # Extract the class-wise metrics
    class_wise_metrics = metrics['class_wise_metrics']
    
    # Create a DataFrame for class-wise metrics
    class_metrics_list = []
    for class_label, metrics in class_wise_metrics.items():
        if class_label not in ['accuracy', 'macro avg', 'weighted avg']:
            class_metrics_list.append({
                'class_label': class_label,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1-score': metrics['f1-score'],
                'support': metrics['support']
            })

    class_metrics_df = pd.DataFrame(class_metrics_list)
    
    # Save class-wise metrics to CSV
    class_metrics_df.to_csv(class_metrics_file, index=False)
    print(f'Class-wise metrics saved to {class_metrics_file}')
    # print(metrics.keys())
