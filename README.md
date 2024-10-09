# Company Category Prediction Using Transfer Learning and Fine-Tuning

## Overview
This project leverages **transfer learning** by fine-tuning a pre-trained **BERT (Bidirectional Encoder Representations from Transformers)** model to predict the business category of companies based on their website’s `meta_description`. The model is trained on a custom dataset and optimized for multi-class classification.

## Transfer Learning and Fine-Tuning
### Transfer Learning
- The `bert-base-uncased` model, pre-trained on a large corpus of English text, was used to leverage its learned language representations.
   
### Fine-Tuning
- The pre-trained BERT model was further **fine-tuned** on a specific task of **company category classification** using the `meta_description` field. Fine-tuning adjusts the model weights slightly to adapt it to this specialized task, resulting in better performance and faster training than training from scratch.

## Methodology
### 1. **Data Preprocessing**
- **Text Cleaning**: Cleaned and standardized website text data, removing unnecessary characters.
- **Non-English Text Removal**: A custom algorithm was developed to detect and exclude non-English website descriptions to ensure consistent training data.
   
### 2. **Model Building**
- **Transfer Learning with BERT**: The `bert-base-uncased` model from Hugging Face’s library was utilized for the task.
- **Fine-Tuning**: The model was fine-tuned specifically for company category prediction by training on a dataset of company `meta_description` text.
- **Class Imbalance Handling**: Stratified split was employed to address class imbalance in the dataset, ensuring even distribution during training.

### 3. **Performance Metrics**
- High performance was achieved through fine-tuning, with an overall accuracy of **89.5%** on the test set.

## Model Details
- **Model**: The `bert-base-uncased` model from Hugging Face’s Transformers library.
- **Transfer Learning**: The BERT model was pre-trained on general English text data and later fine-tuned on this specific classification task.
- **Optimizer**: AdamW with weight decay, combined with a linear learning rate scheduler.
- **Evaluation Metrics**: Precision, Recall, F1-score across multiple classes, with detailed confusion matrix analysis.

## Installation
Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/LefterisKyriazanos/TransferLearning-BERT-CompanyClassifier.git
cd TransferLearning-BERT-CompanyClassifier
pip install -r requirements.txt
```

## Usage
To fine-tune the BERT model for category prediction:  

```bash
python3 src/scripts/modelling/bert_model_training.py
```

To evaluate the fine-tuned model:  

```bash
python src/scripts/modelling/evaluate.py
```

## File Structure  

```bash
/data
    ├── cleaned_data.csv
    └── ...
/src
    ├── /modelling
        ├── bert_model_training.py
        └── ...
dir_tree.txt  # Directory tree for code navigation
README.md
requirements.txt
```

## Future Improvements

- Class-specific tuning to optimize performance on underrepresented categories.
- The implementation of data augmentation techniques to improve generalization for minority classes.
