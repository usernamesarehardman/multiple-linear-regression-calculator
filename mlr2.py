import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the dataset
mtcars = pd.read_csv('mtcars.csv')

# Drop car names (if present), ensure numeric data, and drop missing rows
if 'Unnamed: 0' in mtcars.columns:
    mtcars = mtcars.drop(columns=['Unnamed: 0'])
mtcars = mtcars.apply(pd.to_numeric, errors='coerce').dropna()

# Define the dependent variable
y = mtcars['mpg']

# Refine the model by removing non-significant predictors one at a time
X_refined = mtcars[['wt', 'hp']]  # Removed 'qsec' as it has the highest P-value
X_refined = sm.add_constant(X_refined)

# Fit the refined model
refined_model = sm.OLS(y, X_refined).fit()

# Print the refined model summary
print(refined_model.summary())

# Residual analysis for the refined model
fitted_vals_refined = refined_model.fittedvalues
residuals_refined = refined_model.resid

# Residuals vs. Fitted Values plot
plt.scatter(fitted_vals_refined, residuals_refined)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted (Refined Model)')
plt.show()

# Q-Q Plot for the refined model
sm.qqplot(residuals_refined, line='45')
plt.title('Q-Q Plot (Refined Model)')
plt.show()
