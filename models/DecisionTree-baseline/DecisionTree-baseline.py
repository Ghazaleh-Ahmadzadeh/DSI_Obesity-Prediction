# %%
# Import necessary libraries and methods
import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# %%
# Load training baseline and scaled datasets
data_path = "../data/preprocessed/preprocessed_data_20250720_131842"

X_train_base = pd.read_csv(os.path.join(data_path, 'X_train_baseline.csv'))
y_train_base = pd.read_csv(os.path.join(data_path, 'y_train_baseline.csv'))

X_train_scaled = pd.read_csv(os.path.join(data_path, 'X_train_scaled.csv'))
y_train_scaled = pd.read_csv(os.path.join(data_path, 'y_train_scaled.csv'))

# Load baseline and scaled test datasets
X_test_base = pd.read_csv(os.path.join(data_path, 'X_test_baseline.csv'))
y_test_base = pd.read_csv(os.path.join(data_path, 'y_test_baseline.csv'))

X_test_scaled = pd.read_csv(os.path.join(data_path, 'X_test_scaled.csv'))
y_test_scaled = pd.read_csv(os.path.join(data_path, 'y_test_scaled.csv'))

# %%
# Flatten y_train and y_test to be (n_samples, )
y_train_base_reshaped = np.ravel(y_train_base) 
y_test_base_reshaped = np.ravel(y_test_base)

y_train_sc_reshaped = np.ravel(y_train_scaled) 
y_test_sc_reshaped = np.ravel(y_test_scaled)

print(y_train_base_reshaped.shape)
print(y_train_sc_reshaped.shape)
print(y_test_base_reshaped.shape)
print(y_test_sc_reshaped.shape)

# %% [markdown]
# ### Testing Baseline Model with DecisionTree and Baseline Dataset

# %%
# Baseline model with DecisionTree and baseline datasets
dt = DecisionTreeClassifier()
dt.fit(X_train_base, y_train_base_reshaped)

# Evaluate the best model on the test set
y_pred_dt_base = dt.predict(X_test_base)
train_acc_dt_base = accuracy_score(y_train_base_reshaped, dt.predict(X_train_base))
test_acc_dt_base = accuracy_score(y_test_base_reshaped, y_pred_dt_base)

print(f"DT Baseline Train Accuracy: {train_acc_dt_base:.4f}")
print(f"DT Baseline Test Accuracy: {test_acc_dt_base:.4f}")

print("\nDT Baseline Test Set Classification Report:")
print(classification_report(y_test_base, y_pred_dt_base))

# %%
# DecisionTree confusion matrix
cm_dt_base = confusion_matrix(y_test_base_reshaped, y_pred_dt_base)
ConfusionMatrixDisplay(confusion_matrix = cm_dt_base).plot();

# %% [markdown]
# ### Testing Baseline Model with DecisionTree and Scaled Dataset

# %%
# Baseline model with DecisionTree and scaled datasets
dt = DecisionTreeClassifier()
dt.fit(X_train_scaled, y_train_sc_reshaped)

# Evaluate the best model on the test set
y_pred_dt_sc = dt.predict(X_test_scaled)
train_acc_dt_sc = accuracy_score(y_train_sc_reshaped, dt.predict(X_train_scaled))
test_acc_dt_sc = accuracy_score(y_test_sc_reshaped, y_pred_dt_sc)

print(f"DT Scaled Train Accuracy: {train_acc_dt_sc:.4f}")
print(f"DT Scaled Accuracy: {test_acc_dt_sc:.4f}")

print("\nDT Scaled Test Set Classification Report:")
print(classification_report(y_test_sc_reshaped, y_pred_dt_sc))

# %%
# DecisionTree confusion matrix
cm_dt_sc = confusion_matrix(y_test_sc_reshaped, y_pred_dt_sc)
ConfusionMatrixDisplay(confusion_matrix = cm_dt_sc).plot();


