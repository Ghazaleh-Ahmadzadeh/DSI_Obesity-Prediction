#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# !pip install seaborn


# In[3]:


data_raw_path = '../Downloads/'
data_raw_name = 'ObesityDataSet_raw_and_data_sinthetic.csv'
data_filepath = os.path.join(data_raw_path, data_raw_name)


# In[4]:


df = pd.read_csv(data_filepath)
display(df.head())
display(df.shape)


# In[5]:


df.describe()


# In[6]:


df.dtypes


# In[7]:


# Split features by type
categ_features = []
num_features = []

for feature in df.columns.to_list():
    series_feat = df[feature]
    if series_feat.dtype == 'object':
        categ_features.append(feature)
    else:
        num_features.append(feature)


# In[8]:


for feature in categ_features:
    series_feat = df[feature]
    if series_feat.dtype == 'object':
        unique = series_feat.unique()
        print('Feature: {:.<30} no.: {:.<5} values: {}'.format(feature, len(unique), unique))


# In[9]:


# distribution of num features

for feature in num_features:
    fig, ax = plt.subplots()
    ax.hist(df[feature])
    plt.title(feature)


# In[10]:


# correlation among numerical features
fig, ax = plt.subplots()
m = df[num_features].corr().to_numpy()
sns.heatmap(m, annot=True, cmap='coolwarm', fmt=".2f", xticklabels=num_features, yticklabels=num_features)
plt.title('Correlation Matrix of Num Features')
plt.show();


# In[11]:


# bar plots for categorical data
bar_features = df[categ_features].value_counts()

for feature in categ_features:
    counts = df[feature].value_counts().to_dict()
    print(counts)
    fig, ax = plt.subplots()
    ax.bar(counts.keys(), counts.values())
    if len(counts.keys())>3:
        plt.xticks(rotation=90)
    else:
        plt.xticks(rotation=45)
    plt.title(feature)




# ## Data processing
# - Scale `Age`, `Height`, and `Weight`, options: scikit learn minmax scaler within the range `(1, 5)`
# - One-hot-encode Gender, family_history_with_overweight, CALC, MTRANS
# - Encode categorical features
# - `Age` can be grouped into categories, e.g., group_a < 30 years
# -  
# 

# In[ ]:





# 

# In[12]:


from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

df_encoded = df.copy()
label_encoder = LabelEncoder()
df_encoded['NObeyesdad'] = label_encoder.fit_transform(df_encoded['NObeyesdad'])

for col in df_encoded.select_dtypes(include=['object']).columns:
    if col != 'NObeyesdad':
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

X = df_encoded.drop('NObeyesdad', axis=1)
y = df_encoded['NObeyesdad']


# In[13]:


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns[indices]

plt.figure(figsize=(12, 6))
sns.barplot(x=importances[indices], y=features, palette='coolwarm')
plt.title("Feature Importance using Random Forest")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()


# In[14]:


plt.figure(figsize=(8, 5))
sns.countplot(data=df, y='NObeyesdad', order=df['NObeyesdad'].value_counts().index, palette='viridis')
plt.title("Class Distribution of Target (NObeyesdad)")
plt.xlabel("Count")
plt.ylabel("Class")
plt.tight_layout()
plt.show()


# In[15]:


plt.figure(figsize=(12, 8))
corr_matrix = df_encoded[num_features + ['NObeyesdad']].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap (Numerical Features)")
plt.tight_layout()
plt.show()


# In[16]:


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.legend(*scatter.legend_elements(), title="Classes")
plt.title("PCA: First Two Principal Components")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:





# In[17]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Checking missing values
missing_summary = df.isnull().sum()
print("Missing Values:\n", missing_summary[missing_summary > 0])

# Defining numerical and categorical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Creating transformers for preprocessing
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combining preprocessing into a single column transformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Applying preprocessing
processed_data = preprocessor.fit_transform(df)

# Converting to DataFrame
from sklearn.compose import make_column_selector as selector
encoded_feature_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
all_features = np.concatenate([numerical_cols, encoded_feature_names])
df_processed = pd.DataFrame(processed_data.toarray() if hasattr(processed_data, "toarray") else processed_data, columns=all_features)

print("\nProcessed Data (first 5 rows):")
print(df_processed.head())


# In[ ]:





# In[ ]:




