# multiple-linear-regression-calculator

## Overview
This Python program performs multiple linear regression analysis on the `mtcars` dataset, focusing on evaluating R² and adjusted R² values, analyzing confidence intervals, detecting and addressing multicollinearity, and diagnosing residuals. The program is designed to guide users through refining a regression model and ensuring it meets statistical assumptions for interpretability and validity.

---

## Features
1. **Initial Model Creation**: Fits a multiple linear regression model with specified predictors and outputs R², adjusted R², and detailed statistical summaries.
2. **Multicollinearity Check**: Calculates Variance Inflation Factors (VIF) for predictors and iteratively removes variables with high multicollinearity.
3. **Confidence Interval Analysis**: Computes 95% confidence intervals for all model coefficients.
4. **Residual Diagnostics**: Generates:
   - Residuals vs. Fitted values plot.
   - Q-Q plot for normality of residuals.
   - Residuals vs. Predictor plots to assess systematic patterns.

---

## Requirements
- Python 3.x
- Libraries:
  - `pandas`
  - `statsmodels`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`

To install required libraries, run:
```bash
pip install pandas statsmodels scikit-learn matplotlib seaborn
```
## Usage
1. **Prepare Dataset**:
   - Download or locate the `mtcars.csv` dataset.
   - Ensure the file path is correct when replacing `'path_to_mtcars.csv'` in the script.

2. **Run the Script**:
   - Save the script as `multiple_regression.py`.
   - Execute the script in a terminal or IDE:
     ```bash
     python MLR.py
     ```

3. **Interpret Results**:
   - Examine the output in the terminal for:
     - Initial and final model summaries.
     - VIF values and removed predictors.
     - Confidence intervals for coefficients.
   - View diagnostic plots for residual analysis.

---

## Customization
- **Predictors**: Modify the `predictors` list in the script to include or exclude variables from the analysis.
- **VIF Threshold**: Adjust the `threshold` variable to a different value (e.g., 10) based on the desired tolerance for multicollinearity.

---

## Output
- **Model Statistics**: Printed summaries include coefficients, p-values, and R² metrics.
- **Diagnostic Plots**:
  - Residuals vs. Fitted values.
  - Normal Q-Q plot for residuals.
  - Residuals vs. individual predictors.

---

## Notes
- Ensure the dataset is cleaned and formatted correctly (e.g., no missing values in predictors or target).
- Use domain knowledge to interpret the final model and identify significant predictors.
