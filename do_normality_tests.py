import pandas as pd
import numpy as np
from normality_tests import normality_tests
import scipy.stats as stats

# Set pandas display options
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Read data
file_path = 'market.csv'
market = pd.read_csv(file_path)

###############
# Create variables
value_1 = 'square_PC_1'

# Calculate the desirable values
dss = pd.DataFrame()
dss[f'{value_1}'] = market[f'{value_1}']
dss[f'log_{value_1}'] = np.log(market[f'{value_1}'] + 5)
dss[f'sqrt_{value_1}'] = np.sqrt(market[f'{value_1}'] + 5)
bcx_target, lam = stats.boxcox(market[f'{value_1}'] + 5)
dss[f'bcx_{value_1}'] = bcx_target

# Check the intermediate result
print(dss)

test_result = normality_tests(dss)
print(test_result)
test_result.to_csv(f'/home/kaarlahti/PycharmProjects/kirovsk_230516/tables/normality_test_{value_1}.csv')
