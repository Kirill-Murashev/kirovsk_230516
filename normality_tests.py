import pandas as pd
import numpy as np
import scipy.stats as stats
from statsmodels.stats.diagnostic import lilliefors, normal_ad

# Set pandas display options
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Read data
file_path = 'kirovsk_230516.csv'
market = pd.read_csv(file_path)
market = market[['exposition_days', 'total_discount', 'price', 'unit_price', 'square_total',
                 'square_living', 'square_kitchen', 'ratio_liv_tot', 'ratio_kit_tot',
                 'floor', 'floors', 'age', 'finishing']]
market['finishing'] = market['finishing'] + 1

###############
# Create variables
value_1 = 'finishing'

# Calculate the desirable values
dss = pd.DataFrame()
dss[f'{value_1}'] = market[f'{value_1}']
dss[f'log_{value_1}'] = np.log(market[f'{value_1}'])
dss[f'sqrt_{value_1}'] = np.sqrt(market[f'{value_1}'])
bcx_target, lam = stats.boxcox(market[f'{value_1}'])
dss[f'bcx_{value_1}'] = bcx_target

# Check the intermediate result
print(dss)


def normality_tests(df):
    results = []
    for column in df:
        data = df[column]
        shapiro_stat, shapiro_p = stats.shapiro(data)
        lilliefors_stat, lilliefors_p = lilliefors(data)
        anderson_stat, critical_values, anderson_p = stats.anderson(data)
        jarque_bera_stat, jarque_bera_p = stats.jarque_bera(data)
        dagostino_stat, dagostino_p = stats.normaltest(data)

        results.append({
            'Shapiro-Wilk Statistic': shapiro_stat, 'Shapiro-Wilk P-value': shapiro_p,
            'Shapiro-Wilk Decision': 'H0 cannot be rejected' if shapiro_p > 0.05 else 'H0 rejected',
            'Lilliefors Statistic': lilliefors_stat, 'Lilliefors P-value': lilliefors_p,
            'Lilliefors Decision': 'H0 cannot be rejected' if lilliefors_p > 0.05 else 'H0 rejected',
            'Anderson-Darling Statistic': anderson_stat, 'Anderson-Darling Critical Value (5%)': critical_values[2],
            'Anderson-Darling Decision': 'H0 cannot be rejected' if anderson_stat < critical_values[
                2] else 'H0 rejected',
            'Jarque-Bera Statistic': jarque_bera_stat, 'Jarque-Bera P-value': jarque_bera_p,
            'Jarque-Bera Decision': 'H0 cannot be rejected' if jarque_bera_p > 0.05 else 'H0 rejected',
            "K-squared D'Agostino Statistic": dagostino_stat, "K-squared D'Agostino P-value": dagostino_p,
            "K-squared D'Agostino Decision": 'H0 cannot be rejected' if dagostino_p > 0.05 else 'H0 rejected',
        })

    result_df = pd.DataFrame(results)
    result_df.index = df.columns  # Use column names as index
    result_df = result_df.applymap(lambda x: f'{x:.5f}' if isinstance(x, float) else x)
    return result_df


test_result = normality_tests(dss)
print(test_result)
test_result.to_csv(f'normality_test_{value_1}.csv')
