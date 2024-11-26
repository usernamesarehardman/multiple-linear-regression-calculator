# multiple-linear-regression-calculator

## Overview
This repository contains two Python scripts: `mlr.py` and `vif.py`, which perform Multiple Linear Regression (MLR) and Variance Inflation Factor (VIF) analysis on the `mtcars` dataset.

## Prerequisites

Before running these scripts, ensure you have the following Python libraries installed:

- `pandas`
- `statsmodels`

You can install the required libraries via pip:

```bash
pip install pandas statsmodels
```

---

# `mlr.py` - Multiple Linear Regression

### Purpose:
This script performs a multiple linear regression analysis on the `mtcars` dataset, using each numeric feature to predict the `mpg` (miles per gallon) dependent variable.

### How it works:
1. Loads the `mtcars` dataset from a CSV file.
2. Cleans the data by dropping any non-numeric columns and rows with missing values.
3. Ensures the dataset contains the `mpg` column as the dependent variable.
4. For each numeric predictor in the dataset (except `mpg`), it fits an ordinary least squares (OLS) regression model.
5. Collects the results, including the adjusted R-squared, p-value, and coefficient for each predictor.
6. Displays the results sorted by adjusted R-squared.

### Output:
The script will output a table of the predictors, sorted by the adjusted R-squared value, with the following columns:
- **Predictor**: The predictor variable's name.
- **Adj_R2**: Adjusted R-squared value of the model.
- **P_Value**: The p-value for the predictor.
- **Coefficient**: The coefficient of the predictor.

---

# `vif.py` - Variance Inflation Factor (VIF) Analysis

### Purpose:
This script calculates the Variance Inflation Factor (VIF) for each predictor in the `mtcars` dataset to detect multicollinearity. It also fits a multiple linear regression model with the best and second-best predictors and displays their adjusted R-squared values.

### How it works:
1. Loads and cleans the `mtcars` dataset (similar to `mlr.py`).
2. Selects the best predictor (based on previous analysis in Step 1, manually set as 'wt' here).
3. Calculates the VIF for each predictor to assess multicollinearity.
4. Fits an OLS regression model using the best predictor and each of the remaining predictors (excluding the best predictor) to determine the second-best predictor.
5. Outputs the VIF values and the adjusted R-squared values for the models.

### Output:
1. **VIF Results**: A table displaying the VIF values for each predictor, which indicates multicollinearity.
2. **Second-best Predictor Model Results**: A table with the predictors, sorted by adjusted R-squared, showing the adjusted R-squared, p-value, and coefficient for each model.
