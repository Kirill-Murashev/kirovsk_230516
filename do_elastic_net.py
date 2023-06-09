from sklearn.model_selection import train_test_split
import pandas as pd
from appr_elastic_net import ElasticNetRegression

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

# Split the data into training and test sets
X = market.drop('unit_price', axis=1)
y = market['unit_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
elastic_net = ElasticNetRegression(alpha=0.1, l1_ratio=0.5)

# Train the model
elastic_net.train(X_train, y_train)

# Print out the coefficients
print(elastic_net.get_coefficients())

# Print out the summary metrics
print(elastic_net.get_summary(X_test, y_test))

# Predict new data
data = object_of_valuation.drop('unit_price', axis=1)
new_predictions = elastic_net.predict(data)

print(new_predictions)

