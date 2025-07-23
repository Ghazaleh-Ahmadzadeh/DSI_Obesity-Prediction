#!/usr/bin/env python
# coding: utf-8

# In[56]:


import os
import pandas as pd
import numpy as np
import optuna
from optuna.integration import OptunaSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report
    )
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
import shap


# In[57]:


data_path = '../../data/preprocessed/preprocessed_data_20250720_131842'
X_train_baseline_path = os.path.join(data_path, 'X_train_baseline.csv')
X_test_baseline_path = os.path.join(data_path, 'X_test_baseline.csv')
y_train_baseline_path = os.path.join(data_path, 'y_train_baseline.csv')
y_test_baseline_path = os.path.join(data_path, 'y_test_baseline.csv')


X_train_scaled_path = os.path.join(data_path, 'X_train_scaled.csv')
X_test_scaled_path = os.path.join(data_path, 'X_test_scaled.csv')
y_train_scaled_path = os.path.join(data_path, 'y_train_scaled.csv')
y_test_scaled_path = os.path.join(data_path, 'y_test_scaled.csv')


# In[58]:


X_train_baseline = pd.read_csv(X_train_baseline_path)
X_test_baseline = pd.read_csv(X_test_baseline_path)
y_train_baseline = pd.read_csv(y_train_baseline_path)
y_test_baseline = pd.read_csv(y_test_baseline_path)


# In[59]:


X_train_scaled = pd.read_csv(X_train_scaled_path)
X_test_scaled = pd.read_csv(X_test_scaled_path)
y_train_scaled = pd.read_csv(y_train_scaled_path)
y_test_scaled = pd.read_csv(y_test_scaled_path)


# In[60]:


display(X_train_baseline.head(3))
display(y_train_baseline.head(3))


# In[61]:


display(X_train_scaled.head(3))
display(y_train_scaled.head(3))


# ## **Baseline**

# In[62]:


# Baseline
baseline_dt = DecisionTreeClassifier()
baseline_dt.fit(X_train_baseline, y_train_baseline)
preds_baseline = baseline_dt.predict(X_test_baseline)


# In[63]:


print('{:>8}: {:.3f}'.format('Accuracy', accuracy_score(y_test_baseline, preds_baseline)))
print('{:>8}: {:.3f}'.format('F1 score', f1_score(y_test_baseline, preds_baseline, average='macro')))
print('{:>8}: {:.3f}'.format('ROC score', roc_auc_score(
    y_test_baseline, baseline_dt.predict_proba(X_test_baseline), multi_class='ovr', average='macro'
    )
    ))
print('{:>8}: {}'.format('Report', classification_report(y_test_baseline, preds_baseline)))


# ## **Random Forest w/optuna**

# In[64]:


def opt_rf(X: pd.DataFrame, y: pd.Series) -> OptunaSearchCV:
    """RF model with Optuna hyperparameter optimization.
    
    :param X: Features DataFrame.
    :param y: Target Series.
    """
    
    estimator = RandomForestClassifier(random_state=42)
    params = {
        'criterion': optuna.distributions.CategoricalDistribution(['gini', 'entropy']),
        'n_estimators': optuna.distributions.IntDistribution(10, 100),
        'max_depth': optuna.distributions.IntDistribution(2, 32),
        }

    optuna_search = OptunaSearchCV(estimator, params, cv=5, scoring='accuracy')
    
    return  optuna_search.fit(X, y)


# In[65]:


_opt_rf = opt_rf(X_train_scaled, y_train_scaled)
best_score_rf, best_params_rf = _opt_rf.best_score_, _opt_rf.best_params_
best_rf = _opt_rf.best_estimator_


# In[66]:


print(best_score_rf)
print(best_params_rf)


# In[67]:


# reoptimize model
preds_rf = best_rf.predict(X_test_scaled)
print('{:>8}: {:.3f}'.format('Accuracy', accuracy_score(y_test_scaled, preds_rf)))
print('{:>8}: {:.3f}'.format('F1 score', f1_score(y_test_scaled, preds_rf, average='macro')))
print('{:>8}: {:.3f}'.format('ROC score', roc_auc_score(y_test_scaled, best_rf.predict_proba(X_test_scaled), multi_class='ovr', average='macro')))
print('{:>8}: {}'.format('Report', classification_report(y_test_scaled, preds_rf)))


# ## **Feature Selection**

# In[68]:


# Feature seleciton
feat_sel = mutual_info_classif(X_train_scaled, y_train_scaled, random_state=42)
feat_sel_df = pd.DataFrame({
    'feature': X_train_scaled.columns,
    'mutual_info': feat_sel
}).sort_values(by='mutual_info', ascending=False)

display(feat_sel_df)


# In[69]:


selected_features = feat_sel_df.loc[feat_sel_df['mutual_info'] > 0.1, 'feature'].tolist()
print(selected_features)


# ## **Random Forest w/CV**

# In[70]:


X_train_scaled_red = X_train_scaled[selected_features]
X_test_scaled_red = X_test_scaled[selected_features]


# In[71]:


_opt_rf = opt_rf(X_train_scaled_red, y_train_scaled)
best_score_rf, best_params_rf = _opt_rf.best_score_, _opt_rf.best_params_
best_rf = _opt_rf.best_estimator_


# In[78]:


best_rf


# In[72]:


# reoptimize model
preds_rf = best_rf.predict(X_test_scaled_red)
print('{:>8}: {:.3f}'.format('Accuracy', accuracy_score(y_test_scaled, preds_rf)))
print('{:>8}: {:.3f}'.format('F1 score', f1_score(y_test_scaled, preds_rf, average='macro')))
print('{:>8}: {:.3f}'.format('ROC score', roc_auc_score(y_test_scaled, best_rf.predict_proba(X_test_scaled_red), multi_class='ovr', average='macro')))
print('{:>8}: {}'.format('Report', classification_report(y_test_scaled, preds_rf)))


# ## **SHAP for feature importance**

# In[ ]:


_X_test_scaled_red = X_test_scaled_red.replace({True: 1, False: 0}).values


# In[ ]:


explainer = shap.TreeExplainer(
    best_rf,
    _X_test_scaled_red,
    feature_names = selected_features
    )

shap_values = explainer(_X_test_scaled_red, check_additivity=False)


# In[ ]:


shap.plots.beeswarm(shap_values)


# In[ ]:




