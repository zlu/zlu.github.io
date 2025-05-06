---
layout: post
title: "To dropna() or not to dropna()"
title_cn: "是否使用 dropna()？"
date: 2025-05-06 07:17 +0800
comments: true
---
{% if page.lang == 'cn' %}

# 是否使用 dropna()？

在数据集中，尤其是大型调查数据集中，缺失值是常见的。是否应用 dropna() 并不总是理想的，这取决于数据的性质和训练的目标。

1. 我们可能希望保留样本量。在一个调查数据集中，多个列可能有缺失数据。删除行可能会显著减少数据集的大小，导致我们错过重要信息。
2. 如果缺失数据与特定子群体有关，删除行可能会使数据集产生偏差。这被称为 **非随机缺失（MNAR）** 或 **随机缺失（MAR）**。这会引入偏差，从而对模型的性能产生负面影响，尤其是在我们有一个不平衡的数据集时，少数类别的行可能会被无意中删除。

填充缺失数据有两种常见的方法。我们称它们为 **填充方法 (Imputation)**：

1. pandas fillna()：它用指定的值填充 DataFrame 或 Series 中的缺失值。具体来说，它用列的常数、均值或中位数值填充数值列，用列的众数或最频繁出现的值填充分类列。
2. sklearn SimpleImputer：它提供了与 pandas fillna() 类似的功能，但有一些技术上的差异。SimpleImputer 是为机器学习管道设计的，以确保在训练集和测试集中进行一致的填充。它使用 “fit” 从训练集中学习参数，然后对训练集/测试集执行 “transform”。而 pandas fillna() 可以直接应用于 DataFrame，并通过修改填充的 “inplace” 来实现。

让我们以著名的 [行为风险因素监测系统（BRFSS）](https://www.cdc.gov/brfss/annual_data/annual_data.htm) 数据集为例。

使用 Pandas fillna():
```python
import pandas as pd

# 示例 DataFrame
df = pd.DataFrame({
    'sleptim1': [7, 6, None, 8],
    'smokday2': ['Every day', None, 'Not at all', 'Some days']
})

# 用中位数填充数值列，用众数填充分类列
df['sleptim1'].fillna(df['sleptim1'].median(), inplace=True)
df['smokday2'].fillna(df['smokday2'].mode()[0], inplace=True)

print(df)
# 输出：
#    sleptim1   smokday2
# 0      7.0  Every day
# 1      6.0  Every day
# 2      7.0  Not at all
# 3      8.0  Some days
```


使用 sklearn SimpleImputer:

```python
from sklearn.impute import SimpleImputer
import pandas as pd

# 示例 DataFrame
df = pd.DataFrame({
    'sleptim1': [7, 6, None, 8],
    'smokday2': ['Every day', None, 'Not at all', 'Some days']
})

# 数值填充器（中位数）
num_imputer = SimpleImputer(strategy='median')
df['sleptim1'] = num_imputer.fit_transform(df[['sleptim1']])

# 分类填充器（最频繁）
cat_imputer = SimpleImputer(strategy='most_frequent')
df['smokday2'] = cat_imputer.fit_transform(df[['smokday2']])

print(df)
# 输出（与 fillna 相同）：
#    sleptim1   smokday2
# 0      7.0  Every day
# 1      6.0  Every day
# 2      7.0  Not at all
# 3      8.0  Some days
```


在实践中，如果仅处理 DataFrame，我们会选择使用 fillna()，因为它更方便。如果想将填充作为管道的一部分，我们会使用 SimpleImputer：


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


{% else %}

Missing values in datasets, especially large survey datasets, are common. It is not always ideal to apply dropna() depending on the nature of the data and the goal of the training.

1. We may want to preserve the sample size. In a survey dataset, multiple columns may have missing data. Dropping rows could significantly reduce the dataset size and cause us to miss important information.
2. Dropping rows can bias the dataset if the missing data is related to specific subgroup. This is known as **Missing Not at Random (MNAR)** or **Missing at Random (MAR)**. This introduces bias that negatively impacts the model's performance, especially when we have an imbalanced dataset where minority class rows are inadvertently dropped.

There are two common ways to fill in missing data. We call them **imputation** methods:

1. pandas fillna(): It fills the missing values in a DataFrame or Series with specified values. Specifically, it fills numeric columns with a constant, mean, or median value of the column. It fills categorical columns with mode or most frequent occurrence.
2. sklearn SimpleImputer: It provides similar functionality to pandas fillna() but with some technical differences. SimpleImputer is designed for machine learning pipelines to ensure consistent imputation across train and test sets. It uses 'fit' to learn the parameters from the training set and then performs 'transform' on the train/test sets. Pandas fillna(), on the other hand, can be applied directly to DataFrame by modifying the imputation 'inplace'.

Let's use the famous [Behavioral Risk Factor Surveillance System (BRFSS)](https://www.cdc.gov/brfss/annual_data/annual_data.htm) dataset as an example.

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


{% endif %}


