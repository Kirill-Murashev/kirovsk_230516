# Import libraries
import pandas as pd

# Set pandas display options
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Read data
file_path = 'kirovsk_230516.csv'
market = pd.read_csv(file_path)


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

elevators = calculate_mean_median_integer(market, 'is_elevators', 'unit_price')
print(elevators)
elevators.to_csv('/home/kaarlahti/PycharmProjects/kirovsk_230516/tables/elevators_mean&median.csv')
