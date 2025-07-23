#!/usr/bin/env python
# coding: utf-8

# In[2]:


!pip install lightgbm

# In[6]:


# 1. Setup & Imports 
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Baseline Model
from sklearn.svm import SVC

# Advanced Model
import lightgbm as lgb

import os
import warnings
warnings.filterwarnings('ignore')


# In[7]:


# 2. Load Latest Preprocessed Data 

# Find the most recent preprocessed data folder
try:
    preprocessed_dir = os.path.join('..', 'data', 'preprocessed')
    latest_folder = sorted([d for d in os.listdir(preprocessed_dir) if os.path.isdir(os.path.join(preprocessed_dir, d))])[-1]
    latest_data_path = os.path.join(preprocessed_dir, latest_folder)
    print(f"Loading data from the most recent folder: '{latest_data_path}'")

    # Load the scaled data sets
    X_train = pd.read_csv(os.path.join(latest_data_path, 'X_train_scaled.csv'))
    y_train = pd.read_csv(os.path.join(latest_data_path, 'y_train_scaled.csv')).squeeze()
    X_test = pd.read_csv(os.path.join(latest_data_path, 'X_test_scaled.csv'))
    y_test = pd.read_csv(os.path.join(latest_data_path, 'y_test_scaled.csv')).squeeze()

except (FileNotFoundError, IndexError):
    print("ERROR: Preprocessed data not found.")
    exit()

print(f"\nData loaded successfully. Training with {len(X_train)} samples.")


# In[8]:


# 3. Baseline Model: Support Vector Classifier (SVC) 
print("\n--- Training Baseline Model: Support Vector Classifier (SVC) ---")

svc_model = SVC(random_state=42)

param_grid_svc = {
    'C': [1, 10, 50],
    'kernel': ['rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Use GridSearchCV with 'accuracy' as the scoring metric
grid_svc = GridSearchCV(svc_model, param_grid_svc, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_svc.fit(X_train, y_train)

# Print the best parameters and score
print(f"\nBest parameters for SVC: {grid_svc.best_params_}")
print(f"Best cross-validated Accuracy: {grid_svc.best_score_:.4f}")

# Evaluate the best model on the test set
y_pred_svc = grid_svc.predict(X_test)
train_accuracy_svc = accuracy_score(y_train, grid_svc.predict(X_train))
test_accuracy_svc = accuracy_score(y_test, y_pred_svc)

print(f"\nSVC Train Accuracy: {train_accuracy_svc:.4f}")
print(f"SVC Test Accuracy: {test_accuracy_svc:.4f}")

print("\nSVC Test Set Classification Report:")
print(classification_report(y_test, y_pred_svc))


# In[4]:


# 4. Advanced Model: LightGBM Classifier 
print("\n\n--- Training Advanced Model: LightGBM Classifier ---")

lgbm_model = lgb.LGBMClassifier(random_state=42)

param_grid_lgbm = {
    'n_estimators': [250, 350],
    'learning_rate': [0.01, 0.05],
    'num_leaves': [20, 31, 45],
    'max_depth': [7, 10]
}

# Use GridSearchCV with 'accuracy' as the scoring metric
grid_lgbm = GridSearchCV(lgbm_model, param_grid_lgbm, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_lgbm.fit(X_train, y_train)

# Print the best parameters and score
print(f"\nBest parameters for LightGBM: {grid_lgbm.best_params_}")
print(f"Best cross-validated Accuracy: {grid_lgbm.best_score_:.4f}")

# Evaluate on the test set
y_pred_lgbm = grid_lgbm.predict(X_test)
train_accuracy_lgbm = accuracy_score(y_train, grid_lgbm.predict(X_train))
test_accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)

print(f"\nLightGBM Train Accuracy: {train_accuracy_lgbm:.4f}")
print(f"LightGBM Test Accuracy: {test_accuracy_lgbm:.4f}")

print("\nLightGBM Test Set Classification Report:")
print(classification_report(y_test, y_pred_lgbm))

