## Dataset Note
Dataset is not included in this repository due to GitHub file size limitations.
Please place the dataset CSV inside the `data/` folder before training.
# Mental Health Detection from Social Media Text using Machine Learning

## Project Overview
This project detects possible **depression / mental health risk** from social media text using **Natural Language Processing (NLP)** and **Machine Learning (ML)**.

## Objective
To classify social media text into:
- **Risk Detected**
- **No Risk Detected**

## Technologies Used
- Python
- Pandas
- NLTK
- Scikit-learn
- Matplotlib
- Streamlit

## Workflow
1. Load dataset
2. Combine title and body text
3. Clean and preprocess text
4. Convert text into TF-IDF vectors
5. Train Logistic Regression model
6. Evaluate performance
7. Save trained model and vectorizer
8. Deploy with Streamlit frontend

## Preprocessing
- Lowercasing
- Punctuation removal
- Stopword removal
- Lemmatization

## Model
- **Feature Extraction:** TF-IDF
- **Classifier:** Logistic Regression

## Results
- **Accuracy:** ~90.60%
- Classification report generated
- Confusion matrix plotted

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt

