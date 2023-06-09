# Import libraries
import pandas as pd
from appr_EN_regression import train_best_elastic_net_regression

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

# Use the function to train an Elastic Net regression model
results = train_best_elastic_net_regression(market, 'unit_price')

model = results['model']
scaler = results['scaler']
metrics_df = results['metrics']
coefficients_df = results['coefficients']

# Save the metrics and coefficients to .csv files
metrics_df.to_csv('elastic_net_metrics.csv')
coefficients_df.to_csv('elastic_net_coefficients.csv')

# To make predictions with the model, you can use:
# Suppose we want to predict the first 5 observations
new_data = object_of_valuation.drop('unit_price', axis=1).head()
new_data_scaled = results["scaler"].transform(new_data)  # Scale the new data using the saved scaler
predictions = results["model"].predict(new_data_scaled)
print("\nPredictions:\n", predictions)