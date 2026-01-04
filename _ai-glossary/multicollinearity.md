---
title: Multicollinearity
---
Multicollinearity is a statistical issue in regression analysis where two or more independent (predictor) variables are highly correlated with each other.

#### What this means

When predictors move together, the model has difficulty separating their individual effects on the dependent variable. As a result, coefficient estimates become unreliable.

#### Why it’s a problem

Multicollinearity does not reduce the overall predictive power of the model much, but it does affect interpretation and inference:
- Regression coefficients become unstable (small data changes → big coefficient changes)
- Standard errors increase, making predictors appear statistically insignificant
- Coefficient signs or magnitudes may be counter-intuitive
- Hard to determine which variable actually matters

#### Simple example

Suppose you regress house price on:
- house_size_m2
- number_of_rooms

These two predictors are strongly correlated. The model struggles to decide whether size or rooms explain the price increase.

### Common causes
- Including redundant variables
- Using derived variables (e.g., total_sales and sales_per_day)
- Polynomial terms without centering (e.g., x and x²)

#### How to detect it
- Correlation matrix (high pairwise correlations)
- Variance Inflation Factor (VIF)
- VIF > 5 (moderate)
- VIF > 10 (severe)
- Large standard errors with nonsignificant coefficients despite good model fit

#### How to fix or mitigate it
- Remove or combine correlated predictors
- Use feature selection
- Apply regularization (Ridge, Lasso)
- Use Principal Component Analysis (PCA)
- Center variables when using polynomial terms

#### One-sentence definition

Multicollinearity occurs when independent variables in a regression model are highly correlated, making coefficient estimates unstable and difficult to interpret.

