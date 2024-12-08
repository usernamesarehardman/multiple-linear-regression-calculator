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

# Use only the selected predictors (based on VIF analysis)
X = mtcars[['wt', 'qsec', 'hp']]
X = sm.add_constant(X)  # Add intercept

# Fit the multiple linear regression model
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())

# Residual analysis
fitted_vals = model.fittedvalues
residuals = model.resid

# Residuals vs. Fitted Values plot
plt.scatter(fitted_vals, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs. Fitted')
plt.show()

# Q-Q Plot
sm.qqplot(residuals, line='45')
plt.title('Q-Q Plot')
plt.show()

