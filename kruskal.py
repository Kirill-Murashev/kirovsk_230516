import numpy as np
import pandas as pd
import scipy.stats as stats

# Set pandas display options
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Read data
file_path = 'kirovsk_230516.csv'
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


def run_kruskal(df, dv, ivs):
    """
    Perform Kruskal-Wallis H test.

    Parameters:
    df (pd.DataFrame): the DataFrame
    dv (str): dependent variable
    ivs (list): list of independent variables

    Returns:
    pd.DataFrame: DataFrame with results
    """

    # Create dataframe for results
    results = pd.DataFrame(columns=['Test', 'Dependent variable', 'Independent variable', 'Statistic', 'p-value'])

    for iv in ivs:
        groups = df[iv].unique()
        data_groups = [df[df[iv] == group][dv] for group in groups]

        # Perform Kruskal-Wallis test
        h, p = stats.kruskal(*data_groups)

        results = results._append({'Test': 'Kruskal-Wallis', 'Dependent variable': dv,
                                  'Independent variable': iv, 'Statistic': h, 'p-value': p}, ignore_index=True)

    return results


kruskal_all = run_kruskal(market, 'unit_price', ['rooms', 'walls_type', 'finishing_type', 'is_elevators', 'epoch'])
print(kruskal_all)
kruskal_all.to_csv('/home/kaarlahti/PycharmProjects/kirovsk_230516/tables/kruskal_walls&finishing&elevators&epoch.csv')
