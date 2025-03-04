# Heart Disease Detection

## Overview
This project focuses on detecting heart disease using a Stacking Classifier that combines multiple machine learning models for improved accuracy. The model analyzes various health parameters and predicts the likelihood of a person having heart disease based on medical attributes.

## Features
- Data preprocessing and feature engineering
- Implementation of Stacking Classifier using:
  - Logistic Regression(Meta Model)
  - XGBoost Classifier
  - Other Base Models
- Model evaluation and performance metrics
- Web-based deployment using Streamlit 

## Project Structure
heart-disease-detection/
│── data/                     # Dataset and preprocessing scripts
│   ├── heart_disease.csv     # Raw dataset
│
│── notebooks/                # Jupyter notebooks for model training
│   ├── model_training.ipynb  # Model training and evaluation
│
│── models/                   # Saved models and checkpoints
│   ├── stacking_model.pkl    # Trained Stacking Classifier model
│
│── app/                      # Streamlit application
│   ├── app.py                # Streamlit frontend for predictions        
│── README.md                 # Project documentation
│── .gitignore                # Ignore unnecessary files

### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- Jupyter Notebook or VS Code
- Streamlit (for deployment)

## Model Training
1. Data Preprocessing: Handling missing values, feature encoding, and normalization.
2. stacking Classifier Implementation:
   - Base Models: XGBoost and other classifiers
   - Meta Model: Logistic Regression
3. Evaluation: Using accuracy, precision, recall, F1-score, and ROC-AUC to compare models.
4. Model Saving: The trained model is saved as `stacking_model.pkl` for deployment.

## Deployment
The trained model is integrated into a Streamlit web application where users can input health parameters and get predictions on heart disease risk. The app uses `predict.py` to load the trained model and make predictions.

## Results
The Stacking Classifier achieved an accuracy of 85%.
