# Import libraries
import numpy as np
import pandas as pd

# Read data
file_path = 'market.csv'
market = pd.read_csv(file_path)

# Add new columns
conditions_1 = [
    (market['is_block'] == True),
    (market['is_panel'] == True),
    (market['is_brick'] == True),
    (market['is_cast_in_place'] == True)
]

choices_1 = [1, 2, 3, 4]

market['walls_type'] = np.select(conditions_1, choices_1, default=np.nan)

conditions_2 = [
    (market['is_without_renovation'] == True),
    (market['is_basic_renovation'] == True),
    (market['is_improved_renovation'] == True),
    (market['is_design_renovation'] == True)
]

choices_2 = [1, 2, 3, 4]

market['finishing_type'] = np.select(conditions_2, choices_2, default=np.nan)


def apply_conditions(value):
    if value > 75:
        return 1
    elif 25 < value <= 75:
        return 2
    else:
        return 3


market['epoch'] = market['age'].apply(apply_conditions)

# Remain only significant variables
market = market[['id', 'latitude', 'longitude', 'unit_price', 'is_isolated', 'is_first',
                 'is_chattels', 'is_nice_view', 'is_view_to_both', 'rooms', 'walls_type',
                 'finishing_type', 'is_elevators', 'epoch', 'square_PC_1']]

print(market)
market.to_csv('market.csv')
