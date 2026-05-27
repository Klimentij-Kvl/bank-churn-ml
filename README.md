# Bank Customer Churn Prediction

End-to-end machine learning system for predicting customer churn probability in retail banking.

Tech stack:
Python • Pandas • Scikit-learn • FastAPI • Docker

## Business Problem

Retail banks lose a significant amount of revenue due to customer churn.
Acquiring a new customer is substantially more expensive than retaining an existing one.

The goal of this project is to build an end-to-end machine learning system capable of predicting customer churn probability based on demographic and behavioral data.

The system can help the marketing and retention teams:

identify high-risk customers,
launch targeted retention campaigns,
reduce churn-related revenue losses,
optimize customer support resources.

The project includes:

data analysis,
feature engineering,
ML model training,
model evaluation,
REST API serving,
containerization with Docker.

## Dataset

Source:
Kaggle Bank Customer Churn Dataset

The dataset contains customer demographic and banking activity information:
- credit score
- geography
- balance
- age
- tenure
- estimated salary
- activity status
- etc.

Target variable:
- Exited (1 = customer left the bank)

## Project Architecture

Client Request → FastAPI Service → Preprocessing Pipeline → ML Model → Prediction Response

## ML Pipeline

1. Data cleaning
2. Exploratory data analysis
3. Feature engineering
4. Train/test split
5. Preprocessing pipeline
6. Model training
7. Hyperparameter tuning
8. Model evaluation
9. Model serialization
10. API deployment