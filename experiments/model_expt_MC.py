# %%
import os
import numpy as np
import pandas as pd
import random

# %% [markdown]
# ### Data Loading and Formatting

# %%
# Load training scaled datasets
data_path = "../data/preprocessed/preprocessed_data_20250720_131842"

X_train = pd.read_csv(os.path.join(data_path, 'X_train_scaled.csv'))
y_train = pd.read_csv(os.path.join(data_path, 'y_train_scaled.csv'))

# Load test datasets
X_test = pd.read_csv(os.path.join(data_path, 'X_test_scaled.csv'))
y_test = pd.read_csv(os.path.join(data_path, 'y_test_scaled.csv'))

# %%
# Check X_train and y_train datasets
X_train.head()

# %%
y_train.head()

# %%
# Flatten y_train and y_test to be (n_samples, )
y_train_reshaped = np.ravel(y_train) 
y_test_reshaped = np.ravel(y_test)
print(y_train_reshaped.shape)
print(y_test_reshaped.shape)

# %% [markdown]
# ### Experiment with K Nearest Neighbors for a baseline model
# 

# %%
# Develop baseline model with KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Hyperparameter tuning for KNN model
knn_param = {'n_neighbors': range(1, 11, 1),
            'metric': ['euclidean', 'manhattan'],
            'weights': ['uniform', 'distance']
}

# Create a random forest classifier
knn = KNeighborsClassifier()

# Use GridSearch CV to find the best hyperparameters
knn_search = GridSearchCV(knn,           
            param_grid = knn_param,  
            cv = 10,
            scoring = 'accuracy')

# Fit the Grid search CV object to the data
knn_search.fit(X_train, y_train_reshaped)

knn_results = knn_search.cv_results_

# %%
# Hyperparameters for best KNN model
best_knn = knn_search.best_estimator_
print('Best hyperparameters:',  knn_search.best_params_)
print('Best CV accuracy:', knn_search.best_score_)

# %%
knn_results = knn_search.cv_results_
knn_search_results = pd.concat([pd.DataFrame(knn_results["params"]),
                               pd.DataFrame(knn_results["mean_test_score"], columns=["Accuracy"])],axis=1)
print(knn_search_results)

# %%
# Generate predictions with the best model
y_test_pred_knn = best_knn.predict(X_test)

test_accuracy_knn = accuracy_score(y_test, y_test_pred_knn)
print("Test Accuracy:", test_accuracy_knn)

# %%

knn_report = classification_report(y_test, y_test_pred_knn)
print(knn_report)

# %%
# Create the confusion matrix
cm_knn = confusion_matrix(y_test, y_test_pred_knn)

ConfusionMatrixDisplay(confusion_matrix=cm_knn).plot();

# %% [markdown]
# **Summary for KNN model**
# |Result|Value|
# |---|---|
# |Best hyperparameters| {'metric': 'manhattan', 'n_neighbors': 4, 'weights': 'distance'}|
# |Best CV accuracy| 0.8767821921668076|
# |Test Accuracy| 0.8794326241134752|

# %% [markdown]
# ### Experiment with Random Forest Classifier for more advanced model

# %%
# Develop baseline model with Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

# Hyperparameter tuning for n_estimators and max_depth
param_dist = {'n_estimators': range(50, 501, 50),
              'max_depth': range(1, 26, 5)}

# Create a random forest classifier
rf = RandomForestClassifier(random_state = 42)

# Use GridSearch CV to find the best hyperparameters
GridCV_search = GridSearchCV(rf,           
                            param_grid = param_dist,  
                            cv = 10,
                            scoring = 'accuracy')

# Fit the random search object to the data
GridCV_search.fit(X_train, y_train_reshaped)

# %%
# Summary of GridCV search during hyperparameter tuning
GridCV_results = GridCV_search.cv_results_
CV_search_results = pd.concat([pd.DataFrame(GridCV_results["params"]),
                               pd.DataFrame(GridCV_results["mean_test_score"], columns=["Accuracy"])],axis=1)
print(CV_search_results)

# %%
# Best model with hyperparameters max_depth = 21 and n_estimators = 450
best_rf = GridCV_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  GridCV_search.best_params_)
print('Best CV accuracy:', GridCV_search.best_score_)

# %%
# Generate predictions with the best model
y_pred = best_rf.predict(X_test)

from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)

# %%
# Calculate test accuracy
test_accuracy = accuracy_score(y_test.values.ravel(), y_pred)
print(f"Test Accuracy: {test_accuracy:.3f}")

# %%
from sklearn.metrics import mean_squared_error as MSE
rmse_test = MSE(y_test, y_pred)**(1/2)
print('Test set RMSE: {:.2f}'.format(rmse_test))

# %%
# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(confusion_matrix=cm).plot();

# %%
# Create a series containing feature importances from the model and feature names from the training data
import matplotlib.pyplot as plt
feature_importance = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

feature_importance.plot(kind='barh', color='blue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Features Importance')
plt.show()

# %% [markdown]
# **Summary for Random Forest Model**
# |Result|Value|
# |---|---|
# |Best hyperparameters| {'max_depth': 21, 'n_estimators': 450}|
# |Best CV accuracy|0.9526134122287967|
# |Test Accuracy| 0.941|


