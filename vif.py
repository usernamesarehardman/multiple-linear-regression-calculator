import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the dataset
mtcars = pd.read_csv('mtcars.csv')

# Drop car names (if present), ensure numeric data, and drop missing rows
if 'Unnamed: 0' in mtcars.columns:
    mtcars = mtcars.drop(columns=['Unnamed: 0'])
mtcars = mtcars.apply(pd.to_numeric, errors='coerce').dropna()

# Define the dependent variable
y = mtcars['mpg']

# Initial predictors
predictors = ['wt', 'qsec', 'cyl', 'hp']

# Iteratively calculate and drop predictors with high VIF
def calculate_vif(X):
    vif_data = pd.DataFrame()
    vif_data['Predictor'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

X = mtcars[predictors]
X = sm.add_constant(X)

while True:
    vif_data = calculate_vif(X)
    print("\nCurrent VIF Data:")
    print(vif_data)
    
    # Check if any VIF is greater than threshold
    max_vif = vif_data['VIF'].iloc[1:].max()  # Exclude intercept
    if max_vif > 5:
        # Drop the predictor with the highest VIF
        drop_predictor = vif_data.loc[vif_data['VIF'] == max_vif, 'Predictor'].values[0]
        print(f"Dropping predictor '{drop_predictor}' due to high VIF ({max_vif:.2f})")
        X = X.drop(columns=[drop_predictor])
    else:
        break

# Final predictors
final_predictors = X.columns.tolist()
print("\nFinal predictors after VIF analysis:")
print(final_predictors)

# Fit final model
model = sm.OLS(y, X).fit()
print("\nFinal Model Summary:")
print(model.summary())

