# Written entirely by ChatGPT

# Import libraries
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv("mtcars.csv")  # Replace with your dataset path
features = ["wt", "qsec", "cyl", "hp"]  # Predictors
target = "mpg"  # Target variable

# Step 2: Fit the Initial Model
X = sm.add_constant(data[features])  # Add constant for intercept
y = data[target]


model = sm.OLS(y, X).fit()
print("Initial Model Summary:")
print(model.summary())

# Step 3: Analyze Multicollinearity Using VIF
vif_data = pd.DataFrame()
vif_data["Feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print("\nVariance Inflation Factors (VIF):")
print(vif_data)



# Step 4: Refine the Model (if necessary)

# Example: Remove "cyl" if its VIF is high
refined_features = ["wt", "qsec", "hp"]  # Exclude "cyl" if multicollinear
X_refined = sm.add_constant(data[refined_features])



model_refined = sm.OLS(y, X_refined).fit()
print("\nRefined Model Summary:")
print(model_refined.summary())



# Step 5: Confidence Intervals for Coefficients
print("\nConfidence Intervals for Refined Model Coefficients:")
print(model_refined.conf_int())

# Step 6: Residual Diagnostics
# Residuals vs Fitted
fitted_values = model_refined.fittedvalues
residuals = model_refined.resid

plt.scatter(fitted_values, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted')
plt.show()

# Q-Q Plot for Normality
sm.qqplot(residuals, line='45', fit=True)
plt.title('Normal Q-Q')
plt.show()

# Residuals vs Leverage

sm.graphics.influence_plot(model_refined, criterion="cooks")
plt.title('Residuals vs Leverage')
plt.show()
