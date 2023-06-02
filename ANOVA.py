# Import libraries
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


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


def calculate_mean_median(df, variables, target_column):
    mean_and_median_df = pd.DataFrame()

    for var in variables:
        mean = df[df[var] == True][target_column].mean()
        median = df[df[var] == True][target_column].median()

        mean_and_median_df = mean_and_median_df._append({'Variable': var, 'Mean': mean,
                                                         'Median': median}, ignore_index=True)

    mean_and_median_df.set_index('Variable', inplace=True)
    return mean_and_median_df


def calculate_mean_median_integer(df, variable, target_column):
    mean_and_median_df = pd.DataFrame()

    for val in df[variable].unique():
        mean = df[df[variable] == val][target_column].mean()
        median = df[df[variable] == val][target_column].median()

        mean_and_median_df = mean_and_median_df._append({'Variable_Value': val, 'Mean': mean,
                                                         'Median': median}, ignore_index=True)

    mean_and_median_df.set_index('Variable_Value', inplace=True)
    return mean_and_median_df


# variables = ['is_block', 'is_panel', 'is_brick', 'is_cast_in_place']
# df_walls = calculate_mean_median(market, variables, 'unit_price')
# print(df_walls)
# df_walls.to_csv('/home/kaarlahti/PycharmProjects/kirovsk_230516/tables/walls_type_mean&median.csv')
#
# variables = ['is_without_renovation', 'is_basic_renovation', 'is_improved_renovation', 'is_design_renovation']
# df_finishing = calculate_mean_median(market, variables, 'unit_price')
# print(df_finishing)
# df_finishing.to_csv('/home/kaarlahti/PycharmProjects/kirovsk_230516/tables/finishing_type_mean&median.csv')
#
# elevators = calculate_mean_median_integer(market, 'is_elevators', 'unit_price')
# print(elevators)
# elevators.to_csv('/home/kaarlahti/PycharmProjects/kirovsk_230516/tables/elevators_mean&median.csv')


def run_anova(df, dv, iv, iv2=None, anova_type='one-way'):
    """
    Run a one-way or two-way ANOVA on a DataFrame, df.
    dv is the dependent variable (a string), and iv and iv2 are independent variables.
    If iv2 is not specified, a one-way ANOVA is run. If iv2 is specified, a two-way ANOVA is run.
    """

    results = pd.DataFrame(columns=['Test', 'Group', 'Statistic', 'p-value'])

    # Check for normality in the groups
    groups = df[iv].unique()
    for group in groups:
        _, p = stats.shapiro(df[dv][df[iv] == group])
        results = results._append({'Test': 'Shapiro-Wilk', 'Group': group,
                                  'Statistic': '_', 'p-value': p}, ignore_index=True)

    # Check for homogeneity of variance
    _, p = stats.levene(*[df[dv][df[iv] == group] for group in groups])
    results = results._append({'Test': 'Levene', 'Group': 'All',
                              'Statistic': '_', 'p-value': p}, ignore_index=True)

    # Run the appropriate ANOVA
    if anova_type.lower() == 'one-way':
        model = ols(f'{dv} ~ C({iv})', data=df).fit()
        aov_table = sm.stats.anova_lm(model, typ=2)
        for index, row in aov_table.iterrows():
            results = results._append({'Test': 'ANOVA', 'Group': index,
                                      'Statistic': row['F'], 'p-value': row['PR(>F)']}, ignore_index=True)

        # perform multiple pairwise comparison (Tukey HSD)
        res = pairwise_tukeyhsd(df[dv], df[iv])
        for group1, group2, mean, lower, upper, reject in zip(res.groupsunique,
                                                              res.groupsunique[1:] + res.groupsunique[:1],
                                                              res.meandiffs, res.confint[:, 0],
                                                              res.confint[:, 1], res.reject):
            results = results._append({'Test': 'Tukey HSD', 'Group': f'{group1} - {group2}',
                                      'Statistic': mean, 'p-value': '_' if reject else 'NS'}, ignore_index=True)

    elif anova_type.lower() == 'two-way':
        model = ols(f'{dv} ~ C({iv}) + C({iv2}) + C({iv}):C({iv2})', data=df).fit()
        aov_table = sm.stats.anova_lm(model, typ=2)
        for index, row in aov_table.iterrows():
            results = results._append({'Test': 'ANOVA', 'Group': index,
                                      'Statistic': row['F'], 'p-value': row['PR(>F)']}, ignore_index=True)

        # perform multiple pairwise comparison (Tukey HSD)
        res = pairwise_tukeyhsd(df[dv], df[iv])
        for group1, group2, mean, lower, upper, reject in zip(res.groupsunique,
                                                              res.groupsunique[1:] + res.groupsunique[:1],
                                                              res.meandiffs, res.confint[:, 0], res.confint[:, 1],
                                                              res.reject):
            results = results.append({'Test': 'Tukey HSD', 'Group': f'{group1} - {group2}',
                                      'Statistic': mean, 'p-value': '_' if reject else 'NS'}, ignore_index=True)

        res2 = pairwise_tukeyhsd(df[dv], df[iv2])
        for group1, group2, mean, lower, upper, reject in zip(res2.groupsunique,
                                                              res2.groupsunique[1:] + res2.groupsunique[:1],
                                                              res2.meandiffs, res2.confint[:, 0], res2.confint[:, 1],
                                                              res2.reject):
            results = results._append({'Test': 'Tukey HSD', 'Group': f'{group1} - {group2}',
                                      'Statistic': mean, 'p-value': '_' if reject else 'NS'}, ignore_index=True)

    else:
        print('Invalid ANOVA type. Choose either "one-way" or "two-way"')

    return results


# anova_walls = run_anova(market, 'unit_price', 'walls_type')
# print(anova_walls)
# anova_walls.to_csv('/home/kaarlahti/PycharmProjects/kirovsk_230516/tables/anova_walls.csv')
#
# anova_finishing = run_anova(market, 'unit_price', 'finishing_type')
# print(anova_finishing)
# anova_finishing.to_csv('/home/kaarlahti/PycharmProjects/kirovsk_230516/tables/anova_finishing.csv')
#
# anova_elevators = run_anova(market, 'unit_price', 'is_elevators')
# print(anova_elevators)
# anova_elevators.to_csv('/home/kaarlahti/PycharmProjects/kirovsk_230516/tables/anova_elevators.csv')

# anova_rooms = run_anova(market, 'unit_price', 'rooms')
# print(anova_rooms)
# anova_rooms.to_csv('/home/kaarlahti/PycharmProjects/kirovsk_230516/tables/anova_rooms.csv')

anova_epoch = run_anova(market, 'unit_price', 'epoch')
print(anova_epoch)
anova_epoch.to_csv('/home/kaarlahti/PycharmProjects/kirovsk_230516/tables/anova_epoch.csv')
