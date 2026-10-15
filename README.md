
# Iris Machine Learning Classification

## Overview
This project explores supervised machine learning
using the Iris flower dataset.

## Technologies
- Python
- scikit-learn
- NumPy
- pandas
- Matplotlib

## Models
1. Logistic Regression
2. Random Forest

## Methodology
- 80/20 stratified train/test split
- Five-fold cross-validation
- Hyperparameter tuning
- Error analysis

## Results
- Logistic Regression (C=10): 96.67% mean CV accuracy
- Random Forest (max_depth=1): 96.67% mean CV accuracy
- Tuned Logistic Regression: 100% on 30 test samples

## Limitations
The dataset is small, and the test set was
previously inspected during experimentation.
Results do not guarantee performance on
future data.

## Lessons Learned
I learned how Logistic Regression and Random Forest models work and how different parameters affect the performance of each model. I gained a better understanding of overfitting and underfitting, specifically how changing the C value in Logistic Regression affects regularization and how changing tree depth in Random Forest can affect generalization.

I also learned how to evaluate and compare models using cross-validation, accuracy, and confusion matrices. One interesting thing I discovered was that while both models achieved a cross-validation accuracy of 96.67%, they misclassified different flower samples. To me, this demonstrates how different machine learning algorithms can achieve the same overall accuracy while making different predictions based on what they learned during training.

