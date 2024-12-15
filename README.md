# Intent Detection 

## Problem statement:

  **for the problem statement view this document [Tifin-test_1.pdf](https://github.com/user-attachments/files/18115710/Tifin-test_1.pdf)**

## Overview:

**The goal of this assignment is to build a machine learning model that accurately predicts the intent of a user based on their input. Intent detection is a critical part of most Conversational AI systems as it helps the virtual agent "understand" user needs and respond appropriately.**

## Problem Formulation : (My Novel Approach)

## How this Problem Is Framed as a Machine Learning Problem:

***1.) first of all i have analysed the problem statement and break down  it into different parts .***

***2.) then analysed the each segement of the problem according to the contextual , awareness , sentement analysis , different core  ML approaches.***

***3.) Afterwords i have searched for the research papaers and references for the model to build upon and also different things similar to the contextual awreness and user sentiments models for that you can visit [here](https://ieeexplore.ieee.org/document/10185036)***

***4.) after i have explored the simialr pretrained approaches for it to determine the things and reproduce the output , and analysed  what's the input and output for it.***


### This is the rough diagram of my approach how i framed and break down  the problem :

![WhatsApp Image 2024-12-12 at 15 15 24_e495756d-min](https://github.com/user-attachments/assets/e1143010-5ca3-4876-b819-6a93fd65f57a)


## Reference model arcitecture for intent anlysis:


![image](https://github.com/user-attachments/assets/720c7629-ca06-446d-9f1a-6f5631fa3604)


### For the approach and detail errors and everything can be found in the report:

[tifin_report (1).pdf](https://github.com/user-attachments/files/18117096/tifin_report.1.pdf)


### Some research paper's referneces :

Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.

Pennington, J., Socher, R., & Manning, C. (2014). GloVe: Global Vectors for Word Representation. Conference on Empirical Methods in Natural Language Processing (EMNLP).

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems. 

***you can also visit the below research and documentation:***

**Hugging Face Transformers Documentation: https://huggingface.co/docs**

**Scikit learn Documentation: https://scikit-learn.org/stable/**

**[machine Learning Research Papaers](https://ieeexplore.ieee.org/abstract/document/8292668/)**


# Possible Formulations and Their Pros/Cons:


## 1. Multiclass Classification:

## Formulation: Each utterance is assigned a single intent label.

### Pros:

***1.) Simpler and more efficient to implement.***

***2.)Most intent detection datasets align with this approach.***

## Cons:

***1.)Cannot handle cases where an utterance belongs to multiple intents.***

## 2. Multilabel Classification:

**Formulation: Each utterance can be assigned multiple intent labels.**

### Pros:

**Flexible and capable of handling real-world overlapping intents.**

### Cons:

***More complex implementation and evaluation.***

## Solution Approach:

## 1. Dataset

**The provided dataset contains user utterances with their corresponding intent labels. which is reffered as  ``sentences`` , ``labels`` Key steps include:**

**Exploratory Data Analysis (EDA):** Understanding label distribution, checking for class imbalance, and preprocessing text data (e.g., removing special characters, stopwords).
                                
**Preprocessing:** Converting text to lowercase, tokenization, and vectorization using methods like TF-IDF or pretrained embeddings.

## 2. Implementation

### Two approaches were implemented:

#### Baseline Model:

**Model:** BERT with TF-IDF features.

# Steps:

**Convert text into numerical features using TF-IDF.**

Train a BERT model on the given dataset with normal parameters.

Evaluate using a validation dataset.

**Initial Accuracy:** The baseline model achieved an accuracy of **40.23%,** which was significantly lower than desired.

## Advanced Model (Pretrained Distil-BERT):

**Model:** Fine-tuned BERT (Bidirectional Encoder Representations from Transformers).

### Steps:

**Tokenized inputs using the BERT tokenizer.**

1.) Train a pretrained BERT model for sequence classification on the dataset.

2.) Fine-tune hyperparameters (learning rate, batch size) to improve performance.

3.) Final Accuracy After Tuning: After extensive hyperparameter tuning, the BERT model achieved an accuracy of **84.123%.**

## 3. Tools and Libraries

**Baseline Model:** Python, Sklearn, Pandas, and Numpy, torch, transformers.

**Advanced Model:** PyTorch, Hugging Face Transformers.

## Results:

**Baseline Model (TF-IDF + BERT)**

***Initial Accuracy:*** ``~40.23%``

**for this less accuracy you can visit the colab notebook [here](https://github.com/Blacksujit/Intent-Detection/blob/606620091f30eb21625e09d5e141e4509770c628/first_approach_colab.ipynb)**

## Key Insights:

**1.) Simple preprocessing and feature engineering produced limited results.**

**2.) The model struggled with complex language semantics.**

### **Advanced Model (Fine-tuned BERT)**

**Final Accuracy:**  ``~84.123%`` 

**for this final accuracy you can download the notebook [here ](https://github.com/Blacksujit/Intent-Detection/blob/92ce6773d268e9068e16affa32db1f73f0b52996/intent_notebooks)**

### Key Insights:

**1.) BERT's contextual embeddings significantly improved performance.**

**2.) Hyperparameter tuning was critical in achieving higher accuracy.**

## In future We can also improve the model according to the following parameters:

### Data Augmentation:

**Expand the dataset using back-translation or paraphrasing to increase diversity.**

### Hyperparameter Tuning:

**Experiment with learning rates, batch sizes, and epochs for better performance.**

### Class Imbalance Handling:

**Use techniques like oversampling minority classes or weighted loss functions to improve predictions on underrepresented intents.**

### Ensemble Models:

**Combine traditional and transformer-based models for robust predictions.**

## Installation and setup:

```
git clone https://github.com/Blacksujit/Intent-Detection
```

```visit the pretrained model notebook:``` [here](https://github.com/Blacksujit/Intent-Detection/blob/606620091f30eb21625e09d5e141e4509770c628/intent_notebooks)

```run the notebook```

***save the pretrained model in the same directory as the path ,``pretrained_model.pth``***

```you can now use the saved model for predictions```

## Conclusion:


**1.) using this approaches we have figured out how we can obtain the maximum prediction accuracy from the model.**

**2.) we can also use core ML approaches if we have the larger datasets to prepare our own , models , but by doing the hyperparameter tuining we can achieve the maximum accuracy**


