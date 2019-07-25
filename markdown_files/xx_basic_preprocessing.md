# -*- coding: utf-8 -*-
---
jupyter:
  jupytext:
    formats: notebooks//ipynb,markdown_files//md,python_scripts//py:percent
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Introduction to scikit-learn: basic preprocessing for basic model fitting

In this lecture note, we will aim at introducing:
* the difference between numerical and categorical variables;
* the importance of scaling numerical variables;
* the way to encode categorical variables;
* combine different preprocessing on different type of data;
* evaluate the performance of a model via cross-validation.


## Introduce the dataset

To this aim, we will use data from the 1985 "Current Population Survey"
(CPS). The goal with this data is to regress wages from heterogeneous data
such as age, experience, education, family information, etc.

Let's first load the data located in the `datasets` folder.

```python deletable=true editable=true
import os
import time
import pandas as pd

df = pd.read_csv(os.path.join('datasets', 'cps_85_wages.csv'))
```

We can quickly have a look at the head of the dataframe to check the type
of available data.

```python deletable=true editable=true
print(df.head())
```

The target in our study will be the "WAGE" columns while we will use the
other columns to fit a model

```python deletable=true editable=true
target_name = "WAGE"
target = df[target_name].to_numpy()
data = df.drop(columns=target_name)
```

We can check the number of samples and the number of features available in
the dataset

```python deletable=true editable=true
print(
    f"The dataset contains {data.shape[0]} samples and {data.shape[1]} "
    "features"
)
```

## Work with numerical data

The most intuitive type of data in machine learning which can (almost)
directly be used in machine learning are known as numerical data. We can
quickly have a look at such data by selecting the subset of columns from
the original data.

```python deletable=true editable=true
print(data.columns)
numerical_columns = ['AGE', 'EDUCATION', 'EXPERIENCE']
```

We will use this subset of data to fit linear regressor to infer the wage

```python deletable=true editable=true
data_numeric = data[numerical_columns]
```

When building a machine learning model, it is important to leave out a
subset of the data which we can use later to evaluate the trained model.
The data used to fit a model a called training data while the one used to
assess a model are called testing data.

Scikit-learn provides an helper function `train_test_split` which will
split the dataset into a training and a testing set. It will ensure that
the data are shuffled before splitting the data.

```python deletable=true editable=true
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
```

We will build a Support Vector Machine (SVM) which is a linear model. The
`fit` method is called to train the data and only the training data should
be given for this purpose.
To evaluate our model, we can use the method `score`. It will compute the
coefficient of determination R2 when dealing with a regression problem.

In addition, we checking the time required to train the model and internally
check the number of iterations done by the solver to find a solution.
```python deletable=true editable=true
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
```

We should not the `ConvergenceWarning` which inform us that our model stopped
learning since it reaches the maximum number of iterations allowed by the
user. This could potentially be detrimental for the model accuracy. We can
follow the (bad) advice given in the warning message and increase the maximum
number of iterations allowed.

```python deletable=true editable=true
model = LinearSVR(max_iter=50000)
start = time.time()
model.fit(data_train, target_train)
elapsed_time = time.time() - start
print(
    f"The R2 score using a {model.__class__.__name__} is "
    f"{model.score(data_test, target_test):.2f} with a fitting time of "
    f"{elapsed_time:.3f} seconds in {model.n_iter_} iterations"
)
```

We can observe an increase in performance add the cost of a longer training.
Instead of increasing the number of iterations, we could instead know a bit
more about the SVR model and known that it is expecting input data to be
scaled before to start training. A range of preprocessing algorithms in
scikit-learn allows to transform the input data before to train a model.
We can easily combine these sequential operation with a scikit-learn
`Pipeline` which will chain the operations and can be used as any other
classifier or regressor. The helper function `make_pipeline` will create
a `Pipeline` by giving the successive transformations to perform.

In our case, we will standardize the data and then train a linear SVR.

```python deletable=true editable=true
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
```

We can see that the training time and the number of iterations is much
shorter while the accuracy is equivalent.

```python
from sklearn.model_selection import cross_val_score

score = cross_val_score(model, data_numeric, target)
print(f"The R2 score is: {score.mean():.2f} +- {score.std():.2f}")
print(score)
```

```python
categorical_columns = [
    'SOUTH', 'SEX', 'UNION', 'RACE', 'OCCUPATION', 'SECTOR', 'MARR'
]
data_categorical = data[categorical_columns]
```

```python
from sklearn.preprocessing import OrdinalEncoder

print(data_categorical.head())
print(f"The datasets is composed of {data_categorical.shape[1]} features")
encoder = OrdinalEncoder()
data_encoded = encoder.fit_transform(data_categorical)

print(f"The dataset encoded contains {data_encoded.shape[1]} features")
print(data_encoded[:5])
```

```python
from sklearn.preprocessing import OneHotEncoder

print(data_categorical.head())
print(f"The datasets is composed of {data_categorical.shape[1]} features")
encoder = OneHotEncoder(sparse=False)
data_encoded = encoder.fit_transform(data_categorical)

print(f"The dataset encoded contains {data_encoded.shape[1]} features")
print(data_encoded[:5])
```

```python
model = make_pipeline(OneHotEncoder(handle_unknown='ignore'), LinearSVR())
score = cross_val_score(model, data_categorical, target)
print(f"The R2 score is: {score.mean():.2f} +- {score.std():.2f}")
print(score)
```


```python
from sklearn.compose import make_column_transformer
from sklearn.linear_model import RidgeCV

binary_encoding_columns = ['MARR', 'SEX', 'SOUTH', 'UNION']
one_hot_encoding_columns = ['OCCUPATION', 'SECTOR', 'RACE']
scaling_columns = ['AGE', 'EDUCATION', 'EXPERIENCE']

preprocessor = make_column_transformer(
    (OrdinalEncoder(), binary_encoding_columns),
    (OneHotEncoder(handle_unknown='ignore'), one_hot_encoding_columns),
    (StandardScaler(), scaling_columns)
)
model = make_pipeline(preprocessor, RidgeCV())
score = cross_val_score(model, data, target)
print(f"The R2 score is: {score.mean():.2f} +- {score.std():.2f}")
print(score)
```


```python

```
