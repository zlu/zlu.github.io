---
layout: post
title: "Python in Finance - Part 1"
date: 2025-06-02
comments: true
tags:
  - python
  - artificial intelligence
  - machine learning
  - finance
  - math
description: "Python in Finance - Part 1"
---

Back in 2018, I got interested in capital market. Understanding the fundamental theories of capital market is critical to accumulation of wealth. I started reading all the classics such as The Intelligent Investor and Security Analysis. In this series of posts, I'd like to share with my reader the journey of understanding financial theories in the context of Python programming language. In the first big part of the posts, we will focus on the linearity aspect of financial models, Capital Asset Pricing Model (CAPM), Arbitrage Pricing Theory (APT) and the linear optimization. Later sections will cover the non-linear models.

Linearity is a simplified version of the real world. It is conceptual easy to understand as well as computationally tractable. It is a good starting point to understand the financial theories.
e theories.

### What is Linearity?

In mathematics, **linearity** refers to relationships that satisfy two key properties:

1. **Additivity**: f(x + y) = f(x) + f(y)
2. **Homogeneity**: f(αx) = αf(x)

In finance, linear relationships manifest as:

- **Portfolio returns** are linear combinations of individual asset returns
- **Risk factors** combine linearly to explain asset returns
- **Optimization problems** can be formulated as linear or quadratic programs

## 1. Capital Asset Pricing Model (CAPM)

### Theoretical Foundation

The **Capital Asset Pricing Model** represents one of the most influential theories in finance. CAPM provides a **linear relationship** between an asset's expected return and its systematic risk. CAPM assumes that the investors are rational amongst a few other factors.

### The CAPM Formula

The **Security Market Line** equation expresses expected return as:

$$ E[R_i] = R_f + \beta_i (E[R_m] - R_f) $$

Where:

- $E[R_i]$ = Expected return of asset i
- $R_f$ = Risk-free rate (e.g., Treasury bill rate)
- $\beta_i$ = Beta of asset i (systematic risk measure)
- $E[R_m]$ = Expected market return
- $(E[R_m] - R_f)$ = Market risk premium

### Understanding Beta

**Beta** measures an asset's **sensitivity to market movements**:

$$\beta_i = \frac{Cov(R_i, R_m)}{Var(R_m)} = \rho_{i,m} \frac{\sigma_i}{\sigma_m}$$

**Interpretation:**

- $\beta = 1$: Asset moves exactly with the market
- $\beta > 1$: Asset is **more volatile** than the market (aggressive)
- $\beta < 1$: Asset is **less volatile** than the market (defensive)
- $\beta < 0$: Asset moves **opposite** to the market (rare, e.g., gold)

### Economic Intuition

CAPM's **linear relationship** captures a fundamental trade-off:

- **Higher systematic risk** (beta) requires **higher expected return**
- **Diversifiable risk** is not compensated (can be eliminated)
- The **market portfolio** is mean-variance efficient
- **Risk premium** is proportional to beta

```python
# Python Implementation
from scipy import stats

# Sample data
stock_returns = [0.065, 0.0265, -0.0593, -0.001, 0.0346]
market_returns = [0.055, -0.09, -0.041, 0.045, 0.022]

# Perform linear regression
beta, alpha, r_value, p_value, std_err = stats.linregress(stock_returns, market_returns)

print(f"Beta: {beta}, Alpha: {alpha}")
```

    Beta: 0.5077431878770808, Alpha: -0.008481900352462384

### CAPM Applications

**1. Cost of Equity Capital**

- $r_e = R_f + \beta_e (R_m - R_f)$
- Used in **DCF valuations** and **WACC calculations**

**2. Performance Evaluation**

- **Jensen's Alpha**: $\alpha = R_p - [R_f + \beta_p(R_m - R_f)]$
- Positive alpha indicates **superior performance**

**3. Portfolio Management**

- **Beta targeting** for desired risk levels
- **Market timing** strategies based on beta adjustments

## 2. Arbitrage Pricing Theory (APT)

### Theoretical Foundation

**Arbitrage Pricing Theory extends CAPM by acknowledging that **multiple factors** drive asset returns. APT is based on the **law of one price** and assumes that **arbitrage opportunities\*\* are quickly eliminated, which means that APT is less restrictive than CAPM.

### APT Assumptions

1. **Asset returns follow a linear factor model**
2. **No arbitrage opportunities** exist
3. **Perfect competition** in capital markets
4. **Investors prefer more wealth to less**

### The Multi-Factor Model

APT expresses returns as a **linear combination** of multiple factors:

$$ R*i = \alpha_i + \beta*{i,1}F*1 + \beta*{i,2}F*2 + \ldots + \beta*{i,k}F_k + \epsilon_i $$

Where:

- $R_i$ = Return on asset i
- $\alpha_i$ = Asset-specific return (should be zero in equilibrium)
- $F_j$ = Factor j (e.g., GDP growth, inflation, interest rates)
- $\beta_{i,j}$ = Sensitivity of asset i to factor j
- $\epsilon_i$ = Idiosyncratic error term

### Economic Intuition

APT's **multi-factor approach** recognizes that:

- **Multiple sources of systematic risk** affect asset returns
- **Factor diversification** can reduce portfolio risk
- **Different assets** have different **factor exposures**
- **Risk premiums** exist for **non-diversifiable factors**

```python
# Python Implementation
import numpy as np
import statsmodels.api as sm

# Generate sample data
num_periods = 9
all_values = np.array([np.random.random(8) for _ in range(num_periods)])
y_values = all_values[:, 0]
x_values = all_values[:, 1:]
x_values = sm.add_constant(x_values)

# Perform OLS regression
results = sm.OLS(y_values, x_values).fit()
print(results.summary())
```

                                OLS Regression Results
    ===============================================================================
    Dep. Variable:                      y   R-squared:                       0.968
    Model:                            OLS   Adj. R-squared:                  0.747
    Method:                 Least Squares   F-statistic:                     4.377
    Date:                Sat, 31 May 2025   Prob (F-statistic):              0.353
    Time:                        15:50:56   Log-Likelihood:                 14.758
    No. Observations:                   9   AIC:                            -13.52
    Df Residuals:                       1   BIC:                            -11.94
    Df Model:                           7
    Covariance Type:            nonrobust
    ===============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         -0.3433      0.213     -1.612      0.353      -3.049       2.362
    x1             0.2754      0.201      1.371      0.401      -2.278       2.828
    x2             0.5616      0.262      2.144      0.278      -2.767       3.890
    x3            -0.2459      0.345     -0.713      0.606      -4.631       4.139
    x4             0.4624      0.248      1.862      0.314      -2.693       3.618
    x5             0.8224      0.349      2.357      0.255      -3.612       5.257
    x6            -1.3773      0.413     -3.332      0.186      -6.629       3.875
    x7             1.1494      0.337      3.411      0.182      -3.133       5.432
    ===============================================================================
    Omnibus:                        0.037   Durbin-Watson:                   2.092
    Prob(Omnibus):                  0.982   Jarque-Bera (JB):                0.268
    Skew:                          -0.022   Prob(JB):                        0.875
    Kurtosis:                       2.156   Cond. No.                         23.6
    ===============================================================================

    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

### APT vs. CAPM Comparison

| Aspect               | CAPM                       | APT                          |
| -------------------- | -------------------------- | ---------------------------- |
| **Factors**          | Single (market)            | Multiple                     |
| **Assumptions**      | Restrictive                | More flexible                |
| **Theoretical Base** | Mean-variance optimization | Arbitrage arguments          |
| **Risk Measures**    | Beta only                  | Multiple betas               |
| **Applications**     | Cost of capital            | Risk management, attribution |

### Practical Applications

**1. Risk Management**

- **Multi-factor risk models** for portfolio monitoring
- **Stress testing** across different economic scenarios
- **Hedging strategies** targeting specific risk factors

**2. Performance Attribution**

- **Decomposing returns** by factor contributions
- **Identifying sources** of outperformance/underperformance
- **Style analysis** of investment strategies

**3. Portfolio Construction**

- **Factor-based investing** (smart beta strategies)
- **Risk budgeting** across multiple dimensions
- **Tactical asset allocation** based on factor views

## The Linearity Advantage: Why These Models Work

### Mathematical Benefits

**1. Superposition Principle**
Linear models satisfy **additivity**: the effect of multiple factors equals the sum of individual effects.

**2. Scalability**
Linear relationships are **scale-invariant**: doubling inputs doubles outputs proportionally.

**3. Computational Efficiency**
Linear systems can be solved using **matrix algebra**, enabling:

- **Real-time portfolio optimization**
- **Large-scale risk calculations**
- **Monte Carlo simulations**

### Economic Benefits

**1. Intuitive Interpretation**

- **Coefficients represent sensitivities**
- **Risk contributions are additive**
- **Marginal effects are constant**

**2. Portfolio Construction**

- **Diversification benefits** through linear combinations
- **Risk budgeting** across factors or assets
- **Rebalancing strategies** based on linear rules

**3. Regulatory Compliance**

- **Value-at-Risk (VaR)** calculations
- **Capital requirements** based on linear risk measures
- **Stress testing** using linear factor models

_Next: Linear Optimization in Finance →_
