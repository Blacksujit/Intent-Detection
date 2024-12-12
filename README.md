# Intent Detection Assignment

## Overview

The goal of this assignment is to build a machine learning model that accurately predicts the intent of a user based on their input. Intent detection is a critical part of most Conversational AI systems as it helps the virtual agent "understand" user needs and respond appropriately.

# Problem Formulation:


## How This Problem Is Framed as a Machine Learning Problem:

This problem is formulated as a supervised text classification task, where the input is a text utterance, and the output is a categorical label representing the user’s intent. The steps to solve this include:

Preprocessing the dataset to clean and tokenize text inputs.

Converting text data into numerical features for machine learning models.

Training a model to predict intent categories from the text inputs.

Evaluating model performance using metrics like accuracy, precision, recall, and F1-score.

# Possible Formulations and Their Pros/Cons:

## 1. Multiclass Classification:

## Formulation: Each utterance is assigned a single intent label.

### Pros:

Simpler and more efficient to implement.

Most intent detection datasets align with this approach.

## Cons:

Cannot handle cases where an utterance belongs to multiple intents.

## 2. Multilabel Classification:

Formulation: Each utterance can be assigned multiple intent labels.

Pros:

Flexible and capable of handling real-world overlapping intents.

Cons:

More complex implementation and evaluation.

# Solution Approach

## 1. Dataset

The provided dataset contains user utterances with their corresponding intent labels. Key steps include:

Exploratory Data Analysis (EDA): Understanding label distribution, checking for class imbalance, and preprocessing text data (e.g., removing special characters, stopwords).

Preprocessing: Converting text to lowercase, tokenization, and vectorization using methods like TF-IDF or pretrained embeddings.

## 2. Implementation

## Two approaches were implemented:

## Baseline Model:

Model: Logistic Regression with TF-IDF features.

Steps:

Convert text into numerical features using TF-IDF.

Train a Logistic Regression model on the training dataset.

Evaluate using a validation dataset.

Initial Accuracy: The baseline model achieved an accuracy of 40.23%, which was significantly lower than desired.

## Advanced Model (Pretrained BERT):

Model: Fine-tuned BERT (Bidirectional Encoder Representations from Transformers).

Steps:

Tokenize inputs using the BERT tokenizer.

Train a pretrained BERT model for sequence classification on the dataset.

Fine-tune hyperparameters (learning rate, batch size) to improve performance.

Final Accuracy After Tuning: After extensive hyperparameter tuning, the BERT model achieved an accuracy of 84.123%.

## 3. Tools and Libraries

Baseline Model: Python, Sklearn, Pandas, and Numpy.

Advanced Model: PyTorch, Hugging Face Transformers.

Results

Baseline Model (TF-IDF + Logistic Regression)

Initial Accuracy: ~40.23%

## Key Insights:

Simple preprocessing and feature engineering produced limited results.

The model struggled with complex language semantics.

Advanced Model (Fine-tuned BERT)

Final Accuracy: ~84.123%

Key Insights:

BERT's contextual embeddings significantly improved performance.

Hyperparameter tuning was critical in achieving higher accuracy.

Improvement Suggestions

## Data Augmentation:

Expand the dataset using back-translation or paraphrasing to increase diversity.

## Hyperparameter Tuning:

Experiment with learning rates, batch sizes, and epochs for better performance.

Class Imbalance Handling:

Use techniques like oversampling minority classes or weighted loss functions to improve predictions on underrepresented intents.

Ensemble Models:

Combine traditional and transformer-based models for robust predictions.

## References

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.

Pennington, J., Socher, R., & Manning, C. (2014). GloVe: Global Vectors for Word Representation. Conference on Empirical Methods in Natural Language Processing (EMNLP).

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems.

Hugging Face Transformers Documentation: https://huggingface.co/docs

Scikit-learn Documentation: https://scikit-learn.org/stable/

