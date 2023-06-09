# Import libraries
import pandas as pd
from appr_GLM import appr_glm_regression_stepwise, appr_plot_glm_diagnostics
import statsmodels.api as sm
import math


# Read and transform data
market = pd.read_csv('market_sig.csv')
market = market[['unit_price', 'is_isolated', 'is_first',	'is_chattels', 'is_nice_view', 'is_view_to_both',	'rooms',
                 'is_block', 'is_panel', 'is_brick',	'is_cast_in_place',	'is_without_renovation',
                 'is_basic_renovation', 'is_improved_renovation',	'is_design_renovation',	'is_pass_elevator',
                 'is_freight_elevator', 'is_ussr_1', 'is_ussr_2', 'is_rf', 'square_PC_1']]

object_of_valuation = pd.read_csv('object_sig.csv')

# Check data
print(market)
print(object_of_valuation)

data = market
dependent_var = 'unit_price'
independent_vars = ['is_isolated', 'is_first',	'is_chattels', 'is_nice_view', 'is_view_to_both',	'rooms',
                    'is_block', 'is_panel', 'is_brick',	'is_cast_in_place',	'is_without_renovation',
                    'is_basic_renovation', 'is_improved_renovation',	'is_design_renovation',	'is_pass_elevator',
                    'is_freight_elevator', 'is_ussr_1', 'is_ussr_2', 'is_rf', 'square_PC_1']
family = sm.families.Gaussian()

results, included_vars = appr_glm_regression_stepwise(data, dependent_var, independent_vars, family)

# Load new_data
new_data = object_of_valuation

print(included_vars)

new_data = new_data[included_vars]

# Reindex new_data according to model_vars
new_data = new_data.reindex(columns=included_vars)

# Add constant to new_data
new_data = sm.add_constant(new_data)
print(new_data)

# Make predictions
predictions = results.predict(new_data)
print(predictions)

# Get the required statistics from the model_results
regression_type = 'GLM'
prediction = predictions[0]
# Get the predicted probabilities
predicted_probs = results.predict()
# Calculate the Pseudo R-squared (Cox and Snell)

# Get the log-likelihood
log_likelihood = results.llf

# Get the number of observations
n = results.nobs

# Calculate the Pseudo R-squared (Cox and Snell)
pseudo_r2_cs = results.pseudo_rsquared(kind='cs')
aic = results.aic
bic = results.bic_llf
num_significant_vars = len(included_vars)

price_modelling = pd.read_csv('price_modelling.csv')

new_row = {
    'regression_type': 'GLM Regression',
    'prediction': prediction,
    'R-squared': pseudo_r2_cs,
    'AIC': aic,
    'BIC': bic,
    'number of significant variables': num_significant_vars
}

# Append the new row to the existing DataFrame
price_modelling = price_modelling._append(new_row, ignore_index=True)

# Print the updated DataFrame
print(price_modelling)

# price_modelling.to_csv('price_modelling.csv')

appr_plot_glm_diagnostics(results)