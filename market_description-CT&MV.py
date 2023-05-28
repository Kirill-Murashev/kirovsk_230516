# import libraries
import pandas as pd
import numpy as np
from scipy.stats import trim_mean, mode, iqr
from scipy.stats import percentileofscore

# Set the display format for float values
pd.set_option('display.float_format', '{:.2f}'.format)

# Read data
file_path = 'kirovsk_230516.csv'
market = pd.read_csv(file_path)
market = market[['exposition_days', 'total_discount', 'price', 'unit_price',
                'rooms', 'square_total', 'square_living', 'square_kitchen',
                'ratio_liv_tot', 'ratio_kit_tot', 'floor', 'floors', 'age']]
print(market)


def central_tendencies(data_frame, variable_names):
    statistics = pd.DataFrame()  # Initialize an empty data frame

    for variable_name in variable_names:
        # Calculate mean
        mean = data_frame[variable_name].mean()

        # Calculate truncated mean (trimming 10% of data from both ends)
        truncated_mean = trim_mean(data_frame[variable_name], 0.1)

        # Calculate interquartile mean
        interquartile_mean = np.mean(data_frame[variable_name][data_frame[variable_name].between(data_frame[variable_name].quantile(0.25), data_frame[variable_name].quantile(0.75))])

        # Calculate midrange
        midrange = (data_frame[variable_name].max() + data_frame[variable_name].min()) / 2

        # Calculate midhinge
        midhinge = (data_frame[variable_name].quantile(0.25) + data_frame[variable_name].quantile(0.75)) / 2

        # Calculate trimean
        trimean = (data_frame[variable_name].quantile(0.25) + 2 * data_frame[variable_name].quantile(0.5) + data_frame[variable_name].quantile(0.75)) / 4

        # Calculate mean of 20th, 50th, and 80th percentiles
        percentiles = np.percentile(data_frame[variable_name], [20, 50, 80])
        percentiles_mean = np.mean(percentiles)

        # Calculate winsorized mean (trimming 10% of data from both ends)
        winsorized_mean = np.mean(np.clip(data_frame[variable_name], data_frame[variable_name].quantile(0.1), data_frame[variable_name].quantile(0.9)))

        # Calculate 1st quartile
        first_quartile = data_frame[variable_name].quantile(0.25)

        # Calculate median
        median = data_frame[variable_name].median()

        # Calculate 3rd quartile
        third_quartile = data_frame[variable_name].quantile(0.75)

        # Calculate mode
        mode_val = mode(data_frame[variable_name])[0][0]

        # Create a dictionary with the statistics
        statistics_dict = {'Variable': variable_name,
                           'Mean': mean,
                           'Truncated Mean': truncated_mean,
                           'Interquartile Mean': interquartile_mean,
                           'Midrange': midrange,
                           'Midhinge': midhinge,
                           'Trimean': trimean,
                           'Mean of 20th, 50th, and 80th percentiles': percentiles_mean,
                           'Winsorized Mean': winsorized_mean,
                           '1st Quartile': first_quartile,
                           'Median': median,
                           '3rd Quartile': third_quartile,
                           'Mode': mode_val}

        # Create a data frame from the dictionary
        statistics_df = pd.DataFrame(statistics_dict, index=[0])

        # Concatenate the statistics data frame with the existing 'statistics' data frame
        statistics = pd.concat([statistics, statistics_df], ignore_index=True)

    return statistics
#
#
# def variance_measures(data_frame, variable_names):
#     statistics = pd.DataFrame()  # Initialize an empty data frame
#
#     for variable_name in variable_names:
#         # Calculate statistics for each variable
#         variable_data = data_frame[variable_name]
#
#         # Calculate the statistics
#         minimum = variable_data.min()
#         maximum = variable_data.max()
#         sampling_variance = variable_data.var()
#         sampling_std_deviation = variable_data.std()
#         data_range = maximum - minimum
#         iqr_value = iqr(variable_data)
#         skewness = variable_data.skew()
#         kurtosis = variable_data.kurtosis()
#         gini_coefficient = (np.abs(variable_data - variable_data.mean())).mean() / (2 * variable_data.mean())
#
#         # Create a dictionary with the statistics
#         statistics_dict = {'Variable': variable_name,
#                            'Minimum': minimum,
#                            'Maximum': maximum,
#                            'Sampling Variance': sampling_variance,
#                            'Sampling Std Deviation': sampling_std_deviation,
#                            'Range': data_range,
#                            'Interquartile Range': iqr_value,
#                            'Skewness': skewness,
#                            'Kurtosis': kurtosis,
#                            'Gini Coefficient': gini_coefficient}
#
#         # Create a data frame from the dictionary
#         statistics_df = pd.DataFrame(statistics_dict, index=[0])
#
#         # Concatenate the statistics data frame with the existing 'statistics' data frame
#         statistics = pd.concat([statistics, statistics_df], ignore_index=True)
#
#     return statistics
#
#
# Applying to data
variable_names = market.columns.tolist()  # Get the list of all variable names
result_ct = central_tendencies(market, variable_names)
result_mv = variance_measures(market, variable_names)
print(result_ct)
print(result_mv)

# Save the result to a CSV file
result_ct.to_csv('statistics_CT.csv', index=False)
result_mv.to_csv('statistics_MV.csv', index=False)


# Load the datasets into pandas DataFrames
object = pd.read_csv('object.csv')
object = object[['rooms', 'square_total', 'square_living', 'square_kitchen',
                'ratio_liv_tot', 'ratio_kit_tot', 'floor', 'floors', 'age']]

# Get the list of variables from the datasets
variables = object.columns.tolist()

# Calculate quantiles for each variable in the second dataset relative to the first dataset
quantiles = {}
for variable in variables:
    first_data = market[variable]
    second_data = object[variable].iloc[0]
    quantiles[variable] = percentileofscore(first_data, second_data) / 100

# Create a DataFrame from the quantiles dictionary
quantiles_df = pd.DataFrame.from_dict(quantiles, orient='index', columns=['Quantile'])

# Save the quantiles to a CSV file
quantiles_df.to_csv('quantiles.csv')
