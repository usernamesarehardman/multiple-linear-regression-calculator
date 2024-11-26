import pandas as pd
import statsmodels.api as sm

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

# Loop through numeric predictors
results = []
predictors = [col for col in mtcars.columns if col != 'mpg' and pd.api.types.is_numeric_dtype(mtcars[col])]

for predictor in predictors:
    # Define the current predictor
    X = mtcars[[predictor]]
    X = sm.add_constant(X)  # Add intercept to the model
    
    # Fit the regression model
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
print(results_df)
