# Import libraries
import pandas as pd
import numpy as np
import scipy.stats as stats

# Set pandas display options
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Read data
file_path = 'kirovsk_230516.csv'
market = pd.read_csv(file_path)

# Add new columns
market['is_pass_elevator'] = np.where((market['is_elevators'] == 1) | (market['is_elevators'] == 2), 1, 0)
market['is_freight_elevator'] = np.where(market['is_elevators'] == 2, 1, 0)
print(market)

for col in market.columns:
    if market[col].dropna().isin([0, 1]).all():
        market[col] = market[col].astype(bool)
print(market.dtypes)

# create new DataFrame with only boolean columns
market_bool = market.select_dtypes(include=[bool])
print(market_bool)
market_bool['unit_price'] = market['unit_price']


def boolean_tests(df, value_col):
    alpha = 0.05
    results = []

    # Get the list of boolean columns
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()

    for i, col in enumerate(bool_cols):
        group1 = df[df[col]][value_col]
        group2 = df[~df[col]][value_col]

        # Student's t-test
        t_stat, t_pval = stats.ttest_ind(group1, group2)
        t_significant = t_pval < alpha

        # Welch's t-test
        welch_stat, welch_pval = stats.ttest_ind(group1, group2, equal_var=False)
        welch_significant = welch_pval < alpha

        # Mann-Whitney U-test
        mw_stat, mw_pval = stats.mannwhitneyu(group1, group2)
        mw_significant = mw_pval < alpha

        # Brunner-Munzel test
        bm_stat, bm_pval = stats.brunnermunzel(group1, group2)
        bm_significant = bm_pval < alpha

        # Fligner-Policello test
        fp_stat, fp_pval = stats.fligner(group1, group2)
        fp_significant = fp_pval < alpha

        results.append({
            'bool_var': col,
            't_stat': t_stat, 't_pval': t_pval, 't_significant': t_significant,
            'welch_stat': welch_stat, 'welch_pval': welch_pval, 'welch_significant': welch_significant,
            'mw_stat': mw_stat, 'mw_pval': mw_pval, 'mw_significant': mw_significant,
            'bm_stat': bm_stat, 'bm_pval': bm_pval, 'bm_significant': bm_significant,
            'fp_stat': fp_stat, 'fp_pval': fp_pval, 'fp_significant': fp_significant,
        })

    results_df = pd.DataFrame(results)
    results_df.set_index('bool_var', inplace=True)

    return results_df


boolean_tests_var = boolean_tests(market_bool, 'unit_price')

# Format all float values to 5 decimal places
boolean_tests_var = boolean_tests_var.applymap(lambda x: f'{x:.5f}' if isinstance(x, float) else x)
boolean_tests_var.to_csv('boolean_tests.csv')
