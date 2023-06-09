# Import libraries
import pandas as pd
from appr_ridge_regression import appr_train_best_ridge_regression


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

# Train the model
ridge_cv, metrics, predict, columns = appr_train_best_ridge_regression(market, 'unit_price')

# Align the columns of the new data with the training data
object_of_valuation = object_of_valuation.reindex(columns=columns, fill_value=0)

# Make predictions with new data
predictions = predict(object_of_valuation)
print(predictions)

# Create a DataFrame for the coefficients
coefficients_df = metrics['coefficients']

# Remove the coefficients from the metrics dictionary
metrics_without_coefficients = {k: v for k, v in metrics.items() if k != 'coefficients'}

# Create a DataFrame for the other metrics
metrics_df = pd.DataFrame(metrics_without_coefficients, index=[0])

# Get the results
print(metrics_df)
print(coefficients_df)
print(len(columns))
