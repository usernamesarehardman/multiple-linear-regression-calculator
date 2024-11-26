import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the dataset
mtcars = pd.read_csv('mtcars.csv')

# Drop car names (if present), ensure numeric data, and drop missing rows
if 'Unnamed: 0' in mtcars.columns:
    mtcars = mtcars.drop(columns=['Unnamed: 0'])
mtcars = mtcars.apply(pd.to_numeric, errors='coerce').dropna()

# Ensure 'mpg' exists in the dataset
if 'mpg' not in mtcars.columns:
    raise ValueError("Dataset must include 'mpg' as the dependent variable.")

# Define the dependent variable
y = mtcars['mpg']

# Step 1: Select the best predictor based on Step 1 results
best_predictor = 'wt'  # Change this based on your Step 1 results

# Step 2: Find the second-best predictor
# We'll exclude the best predictor and check the remaining ones
predictors = [col for col in mtcars.columns if col != 'mpg' and col != best_predictor and pd.api.types.is_numeric_dtype(mtcars[col])]

# Calculate VIF to check multicollinearity
X = mtcars[predictors]
X = sm.add_constant(X)
vif_data = pd.DataFrame()
vif_data["Predictor"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print("\nVIF Results (for multi-collinearity):")
print(vif_data)

# Step 3: Fit models with each remaining predictor and check adjusted R-squared
results = []
for predictor in predictors:
    X = mtcars[[best_predictor, predictor]]
    X = sm.add_constant(X)
    
    # Fit the model
    model = sm.OLS(y, X).fit()
    
    # Append results
    results.append({
        'Predictor': predictor,
        'Adj_R2': model.rsquared_adj,
        'P_Value': model.pvalues.iloc[1] if len(model.pvalues) > 1 else None,
        'Coefficient': model.params.iloc[1] if len(model.params) > 1 else None,
    })

# Convert results to a DataFrame and display
results_df = pd.DataFrame(results).sort_values(by='Adj_R2', ascending=False)  # Sort by adjusted R^2
results_df = results_df.round({'Adj_R2': 4, 'P_Value': 4, 'Coefficient': 4})  # Format values
print("\nSecond-best predictor model results:")
print(results_df)

