# Import libraries
import pandas as pd
import statsmodels.api as sm
from appr_linear_regression import appr_multiple_regression, appr_multiple_regression_auto, \
    appr_plot_residuals, create_summary, preprocess_data, predict_new_data, align_new_data,\
    appr_multiple_regression_auto_cv, visualize_regression_coefficients,\
    visualize_scatter_predicted_vs_actual

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


# # Linear regression
# results = appr_multiple_regression(market, 'unit_price')
# print(results.summary())
#
# # Save summary to csv
# model_summary, params_summary = create_summary(results)
# model_summary.to_csv('/home/kaarlahti/PycharmProjects/kirovsk_230516/regression/LR-model_summary.csv')
# params_summary.to_csv('/home/kaarlahti/PycharmProjects/kirovsk_230516/regression/LR-params_summary.csv')

# Automated linear regression
model_results, model_vars = appr_multiple_regression_auto(market, 'unit_price')
print(model_results.summary())

# # Automated linear regression with cross-validation
# model_results, model_vars = appr_multiple_regression_auto_cv(market, 'unit_price')
# print(model_results.summary())

# # # Visualize
# appr_plot_residuals(model_results)
# visualize_scatter_predicted_vs_actual(model_results, market, 'unit_price')
# visualize_regression_coefficients(model_results)


# Align the new data frame with the training data frame
object_of_valuation = object_of_valuation.reindex(columns=market.drop('unit_price', axis=1).columns, fill_value=0)

# Select only the columns present in the model
object_of_valuation = object_of_valuation[model_vars]

# Add a constant column with all values set to 1 as the first column
object_of_valuation.insert(0, 'const', 1)

# Convert the new data frame to a NumPy array
new_data_array = object_of_valuation.values

# Make a prediction
prediction = model_results.predict(new_data_array)
print(prediction)
prediction_value = prediction[0]

# Create a new DataFrame for result keeping
price_modelling = pd.DataFrame(columns=['regression_type', 'prediction', 'Adj. R-squared', 'AIC', 'BIC',
                                        'number of significant variables'])

# Get the required statistics from the model_results
regression_type = 'Multiple Regression'
result = prediction_value
adj_r_squared = model_results.rsquared_adj
aic = model_results.aic
bic = model_results.bic
num_significant_vars = len(model_vars)

# Add the statistics to the price_modelling DataFrame
price_modelling.loc[0] = [regression_type, result, adj_r_squared, aic, bic, num_significant_vars]

# Print the price_modelling DataFrame
print(price_modelling)

price_modelling.to_csv('price_modelling.csv', index=False)