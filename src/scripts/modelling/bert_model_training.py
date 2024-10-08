import numpy as np
import pandas as pd
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, Trainer, TrainingArguments,  EarlyStoppingCallback
from transformers import AdamW, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
from ..helpers import plot_helper, model_helper
import logging

script_dir = os.path.dirname(__file__)

# Construct the relative path to the log file from the script's directory
log_file_path = os.path.join(script_dir, '../../../logs/ml_model_training.log')
data_modelling_path = os.path.join(script_dir, '../../../data/modelling_data/data_for_modelling.csv')
bert_model_file_path = os.path.join(script_dir, '../../../data/models')

logging_dir_ =  os.path.join(script_dir, '../../../logs/bert_training_logs')
path_to_save_eval_results = os.path.join(script_dir, '../../../data/model_evaluation')

# Configurelogging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
       logging.FileHandler(log_file_path, mode='w', encoding='utf-8', delay=False),
       logging.StreamHandler()
    ]
)

# functions and classes 

def main():
   # Load the dataset from a CSV file
    logging.info(f"Loading dataset from {data_modelling_path}.")
    cleaned_data = pd.read_csv(data_modelling_path)
    logging.info(f"Dataset loaded successfully. Shape of the dataset: {cleaned_data.shape}")

    # Encode labels
    logging.info("Encoding labels for the 'Category' column.")
    y, label_mapping, label_encoder = model_helper.encode_labels(cleaned_data, 'Category')
    logging.info(f"Labels encoded. Number of unique categories: {len(label_mapping)}")

    # Use decode_labels to revert back to original Category Values (if needed)

    # Split into train, validation, and test datasets
    logging.info("Splitting data into training, validation, and test sets using stratified split.")
    X_train, X_val, X_test, y_train, y_val, y_test = model_helper.stratified_split(cleaned_data, y, feature_column='meta_description')
    logging.info(f"Data split completed. Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

    # Load the BERT tokenizer
    logging.info("Loading the BERT tokenizer.")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Calculate the number of tokens in each text
    logging.info("Calculating the number of tokens for each text in the training set.")
    token_lengths = [len(tokenizer(text, add_special_tokens=True)['input_ids']) for text in X_train]

    # Analyze the token length distribution
    logging.info("Plotting token length distribution.")
    plot_helper.plot_token_length_distribution(token_lengths, bins=50)

    # Tokenize the data and convert labels
    logging.info("Tokenizing the data and converting labels for BERT model.")
    train_encodings, val_encodings, test_encodings, train_labels, val_labels, test_labels = model_helper.tokenize_and_convert_labels(
        X_train, X_val, X_test, y_train, y_val, y_test, model_name='bert-base-uncased', max_length=64
    )
    logging.info("Tokenization and label conversion completed.")

    # Create the datasets
    logging.info("Creating datasets for training, validation, and testing.")
    train_dataset = model_helper.MetaDescriptionDataset(train_encodings, train_labels)
    val_dataset = model_helper.MetaDescriptionDataset(val_encodings, val_labels)
    test_dataset = model_helper.MetaDescriptionDataset(test_encodings, test_labels)
    logging.info("Datasets created successfully.")

    # Initialize the model
    logging.info("Initializing the BERT model.")
    model_version = 'v2'
    model = model_helper.initialize_model(model_name='bert-base-uncased', num_labels=len(np.unique(y)))

    # Set training arguments
    logging.info("Setting training arguments for the model.")
    training_args = TrainingArguments(
        output_dir=bert_model_file_path,          
        num_train_epochs=4,  # Use 4 epochs as specified
        per_device_train_batch_size=64,  
        per_device_eval_batch_size=64,   
        learning_rate=3e-5,  # Lower learning rate
        warmup_steps=500,                
        weight_decay=0.02,  # Increased weight decay
        logging_dir=logging_dir_,            
        logging_steps=10,
        eval_strategy="epoch",     
        save_strategy="epoch",           
        load_best_model_at_end=True,     
        metric_for_best_model="precision",  # Use precision for early stopping
        greater_is_better=True,  # Lower validation loss is better
        save_total_limit=3,              
        lr_scheduler_type="linear"  # Use linear scheduler with warmup
    )

    # Define the optimizer and scheduler
    logging.info("Defining optimizer and learning rate scheduler.")
    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=3e-5, weight_decay=0.02)

    total_steps = len(train_dataset) // training_args.per_device_train_batch_size * training_args.num_train_epochs
    # Define the scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=500,  # Warmup steps
        num_training_steps=total_steps  # Total steps
    )

    # Initialize the Trainer 
    logging.info("Initializing the Trainer.")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=model_helper.compute_metrics,
        optimizers=(optimizer, scheduler)  # Pass the optimizer and scheduler here
    )
    # Train the model
    logging.info("Starting model training.")
    train_output = trainer.train()
    logging.info(f"Training completed. Total training steps: {train_output.global_step}")
    logging.info(f"Final training loss: {train_output.training_loss}")

    # Evaluate the model
    logging.info("Evaluating the model on the test dataset.")
    eval_df = model_helper.evaluate_results(trainer, test_dataset)
    eval_df.to_csv(f'{path_to_save_eval_results}/{model_version}_eval_results.csv', index=False)
    logging.info(f"Evaluation results saved to {path_to_save_eval_results}/{model_version}_eval_results.csv")

    # Get predictions on the test dataset
    logging.info("Getting predictions on the test dataset.")
    predictions, labels, metrics = trainer.predict(test_dataset)

    # Log metrics
    logging.info(f"Evaluation metrics: {metrics}")

    # Convert logits to predicted class labels
    predicted_labels = np.argmax(predictions, axis=1)

    # Evaluate model performance
    logging.info("Evaluating model performance with confusion matrix and accuracy.")
    metrics = model_helper.evaluate_model_performance(labels, predicted_labels)
    
    
    model_helper.save_metrics_to_csv(metrics, average_metrics_file=f'{path_to_save_eval_results}/{model_version}_average_metrics.csv', class_metrics_file=f'{path_to_save_eval_results}/{model_version}_class_metrics.csv')

    # Save the confusion matrix to CSV
    conf_matrix_df = pd.DataFrame(metrics['confusion_matrix'])
    conf_matrix_df.to_csv(f'{path_to_save_eval_results}/{model_version}_confusion_matrix.csv', index=False, header=False)
    logging.info(f'{path_to_save_eval_results}/{model_version}_confusion_matrix.csv')
    
    # Plot confusion matrix
    logging.info("Plotting confusion matrix.")
    plot_helper.plot_confusion_matrix(metrics['confusion_matrix'])

    # Save actual vs predicted labels
    logging.info("Saving actual vs predicted labels to CSV.")
    
    # Assuming `labels` is your actual labels and `predicted_labels` is your predictions
    labels = np.array(labels)  # Actual labels
    predicted_labels = np.array(predicted_labels)  # Predicted labels

    # Create a DataFrame
    actual_vs_predicted = pd.DataFrame({
        'Actual Label': model_helper.decode_labels(labels, label_encoder), # convert labels to actual class names
        'Predicted Label': model_helper.decode_labels(predicted_labels, label_encoder)
    })

    # Save the DataFrame to a CSV file
    actual_vs_predicted.to_csv(f'{path_to_save_eval_results}/{model_version}_actual_vs_predicted.csv', index=False)
        
    logging.info(f"Actual vs predicted labels saved to {path_to_save_eval_results}/{model_version}_actual_vs_predicted.csv")
    
if __name__ == "__main__":
    main()