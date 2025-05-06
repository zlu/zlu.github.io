---
layout: post
title: To dropna() or not to dropna()
date: 2025-05-06 07:17 +0800
comments: true
---

Missing values in datasets, especially large survey datasets, are common. It is not always ideal to apply dropna() depending on the nature of the data and the goal of the training.

1. We may want to preserve the sample size. In a survey dataset, multiple columns may have missing data. Dropping rows could significantly reduce the dataset size and cause us to miss important information.
2. Dropping rows can bias the dataset if the missing data is related to specific subgroups. This is known as **Missing Not at Random (MNAR)** or **Missing at Random (MAR)**. This introduces bias that negatively impacts the model's performance, especially when we have an imbalanced dataset where minority class rows are inadvertently dropped.

There are two common ways to fill in missing data. We call them **imputation** methods:

1. pandas fillna(): It fills the missing values in a DataFrame or Series with specified values. Specifically, it fills numeric columns with a constant, mean, or median value of the column. It fills categorical columns with mode or most frequent occurrence.
2. sklearn SimpleImputer: It provides similar functionality to pandas fillna() but with some technical differences. SimpleImputer is designed for machine learning pipelines to ensure consistent imputation across train and test sets. It uses 'fit' to learn the parameters from the training set and then performs 'transform' on the train/test sets. Pandas fillna(), on the other hand, can be applied directly to DataFrame by modifying the imputation 'inplace'.

Let's use the famous [Behavioral Risk Factor Surveillance System (BRFSS)][1] dataset as an example.

With Pandas fillna():
```python
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'sleptim1': [7, 6, None, 8],
    'smokday2': ['Every day', None, 'Not at all', 'Some days']
})

# Impute numerical with median, categorical with mode
df['sleptim1'].fillna(df['sleptim1'].median(), inplace=True)
df['smokday2'].fillna(df['smokday2'].mode()[0], inplace=True)

print(df)
# Output:
#    sleptim1   smokday2
# 0      7.0  Every day
# 1      6.0  Every day
# 2      7.0  Not at all
# 3      8.0  Some days
```

With sklearn SimpleImputer
```python
from sklearn.impute import SimpleImputer
import pandas as pd

# Sample DataFrame
df = pd.DataFrame({
    'sleptim1': [7, 6, None, 8],
    'smokday2': ['Every day', None, 'Not at all', 'Some days']
})

# Numerical imputer (median)
num_imputer = SimpleImputer(strategy='median')
df['sleptim1'] = num_imputer.fit_transform(df[['sleptim1']])

# Categorical imputer (most frequent)
cat_imputer = SimpleImputer(strategy='most_frequent')
df['smokday2'] = cat_imputer.fit_transform(df[['smokday2']])

print(df)
# Output (same as fillna):
#    sleptim1   smokday2
# 0      7.0  Every day
# 1      6.0  Every day
# 2      7.0  Not at all
# 3      8.0  Some days
```

In practice, we would opt to use fillna() for its convinience in dealing only with DataFrame.  If we want to apply imputation as part of the pipeline, we would use SimpleImputer:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', pd.get_dummies)
        ]), categorical_cols)
    ])

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
```

[1]: https://www.cdc.gov/brfss/annual_data/annual_data.htm