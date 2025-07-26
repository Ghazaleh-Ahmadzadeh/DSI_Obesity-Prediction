# %%
# Import necessary libraries and methods
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import optuna
from optuna.integration import OptunaSearchCV
import matplotlib.pyplot as plt
import textwrap
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from sklearn.preprocessing import label_binarize
from itertools import cycle

# %%
# Load training scaled datasets
data_path = "../data/preprocessed/preprocessed_data_20250720_131842"

X_train = pd.read_csv(os.path.join(data_path, 'X_train_scaled.csv'))
y_train = pd.read_csv(os.path.join(data_path, 'y_train_scaled.csv'))

# Load test datasets
X_test = pd.read_csv(os.path.join(data_path, 'X_test_scaled.csv'))
y_test = pd.read_csv(os.path.join(data_path, 'y_test_scaled.csv'))

# %%
# Flatten y_train and y_test to be (n_samples, )
y_train_reshaped = np.ravel(y_train) 
y_test_reshaped = np.ravel(y_test)
print(y_train_reshaped.shape)
print(y_test_reshaped.shape)

# %% [markdown]
# ### I) Experiment with baseline model using Support Vector Classifier (SVC)

# %%
# Baseline pipeline using Support Vector Classifier (SVC) as model
pipe_svc = Pipeline([
    ('svc', SVC(random_state = 42, probability = True, decision_function_shape = 'ovr'))
])

# Define parameter grid search for SVC
param_grid_svc = {
    'svc__C': [1, 10, 50],
    'svc__kernel': ['rbf', 'poly'],
    'svc__gamma': ['scale', 'auto']
}

# GridSearchCV with 'accuracy' as the scoring metric
grid_svc = GridSearchCV(pipe_svc, param_grid_svc, cv = 5,
                        scoring = 'accuracy', n_jobs = -1, 
                        verbose = 1)
grid_svc.fit(X_train, y_train_reshaped)


# %%
# Print the best parameters and score
print(f"\nBest SVC parameters: {grid_svc.best_params_}")
print(f"Best cross-validated Accuracy: {grid_svc.best_score_:.4f}")

# Evaluate the best model on the test set
y_pred_svc = grid_svc.predict(X_test)
train_accuracy_svc = accuracy_score(y_train_reshaped, grid_svc.predict(X_train))
test_accuracy_svc = accuracy_score(y_test_reshaped, y_pred_svc)

print(f"SVC Train Accuracy: {train_accuracy_svc:.4f}")
print(f"SVC Test Accuracy: {test_accuracy_svc:.4f}")

print("\nSVC Test Set Classification Report:")
print(classification_report(y_test, y_pred_svc))

# %%
# Create SVC confusion matrix
cm_svc = confusion_matrix(y_test_reshaped, y_pred_svc)
ConfusionMatrixDisplay(confusion_matrix = cm_svc).plot();

# %%
# Define function to plot ROC curve for different models
class_names = np.unique(y_test)

# Define custom intervals for x and y axes
x_ticks = np.arange(0.0, 0.2, 0.05)
y_ticks = np.arange(0.8, 1.05, 0.05)

def plot_roc_curve(model, X_test, y_test, class_names, focus_on_top_left = True, x_ticks = None, y_ticks = None):
    """
    Function to plot ROC curves for multi-class classification problem.

    Parameters:
    model (sklearn model): Trained model (SVC and GridSearch, SVC and Optuna, SVC and Optuna and Feature Selection)
    X_test (array-like): Test features
    y_test (array-like): True labels from test set
    class_names (list): List of class labels for the classes
    focus_on_top_left (bool): Zoom into the top-left quadrant of the ROC curve (default is True)
    x_ticks (list): Custom ticks for the x-axis (optional)
    y_ticks (list): Custom ticks for the y-axis (optional)
    """
    
    # Predict probabilities from the trained model
    y_prob = model.predict_proba(X_test)

    # Bin the true labels
    y_test_bin = label_binarize(y_test, classes = class_names)
    n_classes = y_prob.shape[1]

    # Calculate ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc_per_class = dict() 
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc_per_class[i] = auc(fpr[i], tpr[i])

    # Calculate micro-average ROC curve
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
    roc_auc_per_class["micro"] = auc(fpr["micro"], tpr["micro"])

    # Calculate macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc_per_class["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(8, 6))

    # ROC curves for micro and macro averages
    plt.plot(fpr["micro"], tpr["micro"], label = f"Micro-average ROC curve (AUC = {roc_auc_per_class['micro']:.2f})", 
             color = "pink", linestyle = ":", linewidth = 2)
    plt.plot(fpr["macro"], tpr["macro"], label = f"Macro-average ROC curve (AUC = {roc_auc_per_class['macro']:.2f})", 
             color = "navy", linestyle = ":", linewidth = 2)

    # Assign colors for each class
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "green", "blue", "purple"])

    # Plot each class ROC curve
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color = color, lw = 2, 
                 label = f"{class_names[i]} ROC (AUC = {roc_auc_per_class[i]:.2f})")

    # Plot diagonal line (no discrimination line)
    plt.plot([0, 1], [0, 1], "k--", lw = 1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    # Zoom on top-left area of plot
    if focus_on_top_left:
        plt.xlim([0.0, 0.2]) 
        plt.ylim([0.8, 1.05])

    # Provide custom ticks for the x and y axes
    if x_ticks is not None:
        plt.xticks(x_ticks)
    if y_ticks is not None:
        plt.yticks(y_ticks)
        
    # Get the name of the model class
    model_name = model.__class__.__name__  
    plt.title(f"ROC Curves for {model_name} (OvR macro & micro avg)")
    plt.legend(loc = "lower right")

    plt.show()

# %%
# Plot ROC curve for SVC and GridSearchCV model
plot_roc_curve(grid_svc, X_test, y_test, class_names, focus_on_top_left = False, x_ticks = None, y_ticks = None)

# %%
# Plot zoomed-in ROC curve for SVC and GridSearchCV model
plot_roc_curve(grid_svc, X_test, y_test, class_names, focus_on_top_left = True, x_ticks = x_ticks, y_ticks = y_ticks)

# %% [markdown]
# ### II) Compare GridSearch CV vs Optuna performance

# %%
# Use same svc pipeline and parameters but with optuna instead of GridSearchCV
pipe_svc = Pipeline([
    ('svc', SVC(random_state = 42, probability = True, decision_function_shape = 'ovr'))
])

# Define parameter grid search for SVC
param_grid_opt = {
    'svc__C': optuna.distributions.IntDistribution(1, 10, 50),
    'svc__kernel': optuna.distributions.CategoricalDistribution(['rbf', 'poly']),
    'svc__gamma': optuna.distributions.CategoricalDistribution(['scale', 'auto']),
}

# OptunaSearchCV with 'accuracy' as the scoring metric
opt_svc = OptunaSearchCV(pipe_svc, param_grid_opt, n_trials = 100,
                         cv = 5, scoring = 'accuracy', n_jobs = -1,
                         verbose = 1)

opt_svc.fit(X_train, y_train_reshaped)

# %%
# Print the best parameters and score
print(f"\nBest SVC Opt parameters: {opt_svc.best_params_}")
print(f"Best cross-validated Accuracy: {opt_svc.best_score_:.4f}")

# Evaluate the best model on the test set
y_pred_opt = opt_svc.predict(X_test)
train_acc_opt = accuracy_score(y_train_reshaped, opt_svc.predict(X_train))
test_acc_opt = accuracy_score(y_test_reshaped, y_pred_opt)

print(f"SVC Opt Train Accuracy: {train_acc_opt:.4f}")
print(f"SVC Opt Test Accuracy: {test_acc_opt:.4f}")

print("\nSVC Opt Test Set Classification Report:")
print(classification_report(y_test, y_pred_opt))

# %%
# Create SVC and Optuna confusion matrix
cm_svc_optuna = confusion_matrix(y_test_reshaped, y_pred_opt)

ConfusionMatrixDisplay(confusion_matrix = cm_svc_optuna).plot();

# %%
# Plot ROC curve for SVC and OptunaSearchCV model
plot_roc_curve(opt_svc, X_test, y_test, class_names, focus_on_top_left = False, x_ticks = None, y_ticks = None)

# %%
# Plot zoomed-in ROC curve for SVC and OptunaSearchCV model
plot_roc_curve(opt_svc, X_test, y_test, class_names, focus_on_top_left = True, x_ticks = x_ticks, y_ticks = y_ticks)

# %% [markdown]
# ### III) Feature Selection Using Scikit-learn mutual_info_classif

# %%
# Feature selection using mutual_info_classif

feat_sel = mutual_info_classif(X_train, y_train_reshaped, random_state = 42)
feat_sel_df = pd.DataFrame({
    'feature': X_train.columns,
    'mutual_info': feat_sel
}).sort_values(by = 'mutual_info', ascending = False)

display(feat_sel_df)

# %%
# Wrap long x labels
import textwrap
wrapped_labels = [textwrap.fill(label, 20) for label in X_train.columns]

plt.figure(figsize = (14, 8))
plt.bar(X_train.columns, feat_sel)
plt.xticks(ticks = range(len(wrapped_labels)),
           labels = wrapped_labels, rotation = 45)
plt.ylabel("Score")
plt.title("Mutual Info Classif Feature Scores")
plt.tight_layout()
plt.show()

# %%
selected_features = feat_sel_df.loc[feat_sel_df['mutual_info'] > 0.1, 'feature'].tolist()
print(selected_features)

# %%
# Create new dataframes for selected features based on mutual info classif
X_train_sel = X_train[selected_features]
X_test_sel = X_test[selected_features]

# Use same svc pipeline and parameters but with optuna instead of GridSearchCV
pipe_svc = Pipeline([
    ('svc', SVC(random_state = 42, probability = True, decision_function_shape = 'ovr'))
])

# Define parameter grid search for SVC
param_grid_opt_mic = {
    'svc__C': optuna.distributions.IntDistribution(1, 10, 50),
    'svc__kernel': optuna.distributions.CategoricalDistribution(['rbf', 'poly']),
    'svc__gamma': optuna.distributions.CategoricalDistribution(['scale', 'auto']),
}

# OptunaSearchCV with 'accuracy' as the scoring metric
opt_mic = OptunaSearchCV(pipe_svc, param_grid_opt_mic, n_trials = 100,
                         cv = 5, scoring = 'accuracy', n_jobs = -1,
                         verbose = 1)

opt_mic.fit(X_train_sel, y_train_reshaped)

# %%
# Print the best parameters and score
print(f"\nBest SVC Opt parameters: {opt_mic.best_params_}")
print(f"Best cross-validated Accuracy: {opt_mic.best_score_:.4f}")

# Evaluate the best model on the test set
y_pred_opt_mic = opt_mic.predict(X_test_sel)
train_acc_opt_mic = accuracy_score(y_train_reshaped, opt_mic.predict(X_train_sel))
test_acc_opt_mic = accuracy_score(y_test_reshaped, y_pred_opt_mic)

print(f"Opt MIC Train Accuracy: {train_acc_opt_mic:.4f}")
print(f"Opt MIC Test Accuracy: {test_acc_opt_mic:.4f}")

print("\nOpt MIC Test Set Classification Report:")
print(classification_report(y_test, y_pred_opt_mic))

# %%
# Create optuna and feature selection confusion matrix
cm_opt_mic = confusion_matrix(y_test_reshaped, y_pred_opt_mic)

ConfusionMatrixDisplay(confusion_matrix = cm_opt_mic).plot();

# %%
# Plot ROC curve for SVC, OptunaSearchCV, and multi inform classif feature selection model
plot_roc_curve(opt_mic, X_test_sel, y_test, class_names, focus_on_top_left = False, x_ticks = None, y_ticks = None)

# %%
# Plot zoomed-in ROC curve for SVC, OptunaSearchCV, and multi inform classif feature selection model
plot_roc_curve(opt_mic, X_test_sel, y_test, class_names, focus_on_top_left = True, x_ticks = x_ticks, y_ticks = y_ticks)

# %% [markdown]
# ### IV) Feature selection with SelectKBest

# %%
# Define pipeline with SelectKBest feature selection before Optuna search
pipe_opt_kbest = Pipeline([
    ('select', SelectKBest(score_func = f_classif)),  # Feature selection
    ('svc', SVC(random_state = 42, probability = True, decision_function_shape = 'ovr'))
])

# Define optuna search parameter space
params_opt_kbest = {
    'select__k': optuna.distributions.IntDistribution(6, 20, 1),
    'svc__C': optuna.distributions.IntDistribution(1, 10, 50),
    'svc__kernel': optuna.distributions.CategoricalDistribution(['rbf', 'poly']),
    'svc__gamma': optuna.distributions.CategoricalDistribution(['scale', 'auto'])
    }

opt_kbest = OptunaSearchCV(pipe_opt_kbest, params_opt_kbest, 
                           n_trials = 100, cv = 5, n_jobs = -1, 
                           verbose = 2)
opt_kbest.fit(X_train, y_train_reshaped)

# %%
# Print the best parameters and score
print(f"Best parameters for Opt KBest: {opt_kbest.best_params_}")
print(f"Best cross-validated Accuracy: {opt_kbest.best_score_:.4f}")

# Evaluate the best model on the test set
y_pred_opt_kbest = opt_kbest.predict(X_test)
train_acc_opt_kbest = accuracy_score(y_train_reshaped, opt_kbest.predict(X_train))
test_acc_opt_kbest = accuracy_score(y_test_reshaped, y_pred_opt_kbest)

print(f"\nOpt KBest Train Accuracy: {train_acc_opt_kbest:.4f}")
print(f"Opt KBest Test Accuracy: {test_acc_opt_kbest:.4f}")

print("\nOpt KBest Test Set Classification Report:")
print(classification_report(y_test, y_pred_opt_kbest))

# %%
# Determine which features were selected using SelectKBest
sel = opt_kbest.best_estimator_.named_steps['select']  
mask = sel.get_support()
kbest_feats = X_train.columns[mask]
print("Selected features:", list(kbest_feats))

kbest = opt_kbest.best_estimator_.named_steps['select']
kbest_mask = kbest.get_support()  # Boolean mask
kbest_indices = kbest.get_support(indices=True)  # Feature indices

# 3. Get feature scores and p-values
feature_scores = kbest.scores_
p_values = kbest.pvalues_

# 4. Show selected feature scores
print("Selected feature indices:", kbest_indices)
print("Feature scores:", feature_scores[kbest_indices])
print("P-values:", p_values[kbest_indices])

# %%
# Show features and score by SelectKBest 
feature_names = [f"Feature {i}" for i in range(len(feature_scores))]

# Convert to array if needed
feature_scores = np.array(feature_scores)

plt.figure(figsize = (14, 8))
plt.bar(feature_names, feature_scores)
plt.xticks(rotation = 45)
plt.ylabel("Score")
plt.title("SelectKBest Feature Scores")
plt.tight_layout()
plt.show()

# %%
# Create Optuna and KBestconfusion matrix
cm_opt_kbest = confusion_matrix(y_test_reshaped, y_pred_opt_kbest)

ConfusionMatrixDisplay(confusion_matrix = cm_opt_kbest).plot();

# %%
# Plot ROC curve for SVC, OptunaSearchCV, and select KBest feature selection model
plot_roc_curve(opt_kbest, X_test, y_test, class_names, focus_on_top_left = False, x_ticks = None, y_ticks = None)

# %%
# Plot zoomed-in ROC curve for SVC, OptunaSearchCV, and select KBest feature selection model
plot_roc_curve(opt_kbest, X_test, y_test, class_names, focus_on_top_left = True, x_ticks = x_ticks, y_ticks = y_ticks)


