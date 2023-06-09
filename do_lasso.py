# Import libraries
import pandas as pd
from appr_lasso_regression import appr_train_best_lasso_regression


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
model, metrics, predict, features = appr_train_best_lasso_regression(market, 'unit_price')

# Disable the scientific notation and set the precision to 5 digits after the dot
pd.set_option('display.float_format', '{:.5f}'.format)

# Print the coefficients
coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': metrics['Coefficients']
})
print("\nCoefficients:\n")
print(coefficients)

print(metrics)

# Predict the target for the new observation
prediction = predict(object_of_valuation)
print('Prediction for the new observation:', prediction)

