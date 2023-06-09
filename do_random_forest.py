# Import libraries
import pandas as pd
from appr_random_forest import train_random_forest, predict_new_data_rf

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

# Train model
model = train_random_forest(market, 'unit_price', params={
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
})

# Predict the value
data = object_of_valuation.drop('unit_price', axis=1)
predictions = predict_new_data_rf(model, data)
print(predictions)
