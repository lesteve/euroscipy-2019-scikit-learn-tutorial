# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,markdown_files//md,python_scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Introduction to scikit-learn: basic preprocessing for basic model fitting

import os
import time
import pandas as pd

df = pd.read_csv(os.path.join('datasets', 'cps_85_wages.csv'))
df.head()

# %%
print(df.columns)

# %%
target_name = "WAGE"
target = df[target_name].to_numpy()
data = df.drop(columns=target_name)

# %%
print(
    f"The dataset contains {data.shape[0]} samples and {data.shape[1]} "
    "features"
)

#%%
print(data.columns)
numerical_columns = ['AGE', 'EDUCATION', 'EXPERIENCE']

data_numeric = data[numerical_columns]

# %%
from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, target, random_state=42
)

print(
    f"The training dataset contains {data_train.shape[0]} samples and "
    f"{data_train.shape[1]} features"
)
print(
    f"The testing dataset contains {data_test.shape[0]} samples and "
    f"{data_test.shape[1]} features"
)

#%%
from sklearn.svm import LinearSVR

model = LinearSVR()
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start
print(
    f"The R2 score using a {model.__class__.__name__} is "
    f"{model.score(data_test, target_test):.2f} with a fitting time of "
    f"{elapsed_time:.3f} seconds in {model.n_iter_} iterations"
)

#%%
model = LinearSVR(max_iter=50000)
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start
print(
    f"The R2 score using a {model.__class__.__name__} is "
    f"{model.score(data_test, target_test):.2f} with a fitting time of "
    f"{elapsed_time:.3f} seconds in {model.n_iter_} iterations"
)

#%%
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), LinearSVR())
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start
print(
    f"The R2 score using a {model.__class__.__name__} is "
    f"{model.score(data_test, target_test):.2f} with a fitting time of "
    f"{elapsed_time:.3f} seconds in {model[-1].n_iter_} iterations"
)

#%%
from sklearn.model_selection import cross_val_score

score = cross_val_score(model, data_numeric, target)
print(f"The R2 score is: {score.mean():.2f} +- {score.std():.2f}")
print(score)

#%%
categorical_columns = [
    'SOUTH', 'SEX', 'UNION', 'RACE', 'OCCUPATION', 'SECTOR', 'MARR'
]
data_categorical = data[categorical_columns]

#%%
from sklearn.preprocessing import OrdinalEncoder

print(data_categorical.head())
print(f"The datasets is composed of {data_categorical.shape[1]} features")
encoder = OrdinalEncoder()
data_encoded = encoder.fit_transform(data_categorical)

print(f"The dataset encoded contains {data_encoded.shape[1]} features")
print(data_encoded[:5])

#%%
from sklearn.preprocessing import OneHotEncoder

print(data_categorical.head())
print(f"The datasets is composed of {data_categorical.shape[1]} features")
encoder = OneHotEncoder(sparse=False)
data_encoded = encoder.fit_transform(data_categorical)

print(f"The dataset encoded contains {data_encoded.shape[1]} features")
print(data_encoded[:5])

#%%
model = make_pipeline(OneHotEncoder(handle_unknown='ignore'), LinearSVR())
score = cross_val_score(model, data_categorical, target)
print(f"The R2 score is: {score.mean():.2f} +- {score.std():.2f}")
print(score)


#%%
from sklearn.compose import make_column_transformer

preprocessor = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), categorical_columns),
    (StandardScaler(), numerical_columns)
)
model = make_pipeline(preprocessor, LinearSVR())
score = cross_val_score(model, data, target)
print(f"The R2 score is: {score.mean():.2f} +- {score.std():.2f}")
print(score)


#%%
