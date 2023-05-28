# import libraries
import pandas as pd
import numpy as np
from scipy.stats import trim_mean, iqr, skew, kurtosis
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


# Define a function to calculate the Gini coefficient
# def gini(x):
#     x = np.array(x)
#     mad = np.abs(np.subtract.outer(x, x)).mean()  # mean absolute difference
#     rmad = mad / np.mean(x)  # relative mean absolute difference
#     g = 0.5 * rmad  # Gini coefficient
#     return g


def bootstrap_samples(data, n_bootstrap_samples=1000):
    n = len(data)
    bootstrap_samples = np.random.choice(data, (n_bootstrap_samples, n), replace=True)
    return bootstrap_samples


# def bootstrap_central_tendencies(data_frame, variable_names, n_bootstrap_samples=1000):
#     statistics = pd.DataFrame()  # Initialize an empty data frame
#
#     for variable_name in variable_names:
#         data = data_frame[variable_name]
#         bs_samples = bootstrap_samples(data, n_bootstrap_samples)
#
#         # Calculate mean
#         means = bs_samples.mean(axis=1)
#         mean = means.mean()
#
#         # Calculate truncated mean (trimming 10% of data from both ends)
#         truncated_means = [trim_mean(bs_sample, 0.1) for bs_sample in bs_samples]
#         truncated_mean = np.mean(truncated_means)
#
#         # Calculate interquartile mean
#         interquartile_means = [np.mean(bs_sample[(bs_sample >= np.percentile(bs_sample, 25)) & (bs_sample <= np.percentile(bs_sample, 75))]) for bs_sample in bs_samples]
#         interquartile_mean = np.mean(interquartile_means)
#
#         # Calculate midrange
#         midranges = [(bs_sample.max() + bs_sample.min()) / 2 for bs_sample in bs_samples]
#         midrange = np.mean(midranges)
#
#         # Calculate midhinge
#         midhinges = [(np.percentile(bs_sample, 25) + np.percentile(bs_sample, 75)) / 2 for bs_sample in bs_samples]
#         midhinge = np.mean(midhinges)
#
#         # Calculate trimean
#         trimeans = [(np.percentile(bs_sample, 25) + 2 * np.median(bs_sample) + np.percentile(bs_sample, 75)) / 4 for bs_sample in bs_samples]
#         trimean = np.mean(trimeans)
#
#         # Calculate mean of 20th, 50th, and 80th percentiles
#         percentiles_means = [np.mean(np.percentile(bs_sample, [20, 50, 80])) for bs_sample in bs_samples]
#         percentiles_mean = np.mean(percentiles_means)
#
#         # Calculate winsorized mean (trimming 10% of data from both ends)
#         winsorized_means = [np.mean(np.clip(bs_sample, np.percentile(bs_sample, 10), np.percentile(bs_sample, 90))) for bs_sample in bs_samples]
#         winsorized_mean = np.mean(winsorized_means)
#
#         # Calculate 1st quartile
#         first_quartiles = [np.percentile(bs_sample, 25) for bs_sample in bs_samples]
#         first_quartile = np.mean(first_quartiles)
#
#         # Calculate median
#         medians = [np.median(bs_sample) for bs_sample in bs_samples]
#         median = np.mean(medians)
#
#         # Calculate 3rd quartile
#         third_quartiles = [np.percentile(bs_sample, 75) for bs_sample in bs_samples]
#         third_quartile = np.mean(third_quartiles)
#
#
#         # Create a dictionary with the statistics
#         statistics_dict = {'Variable': variable_name,
#                            'Mean': mean,
#                            'Truncated Mean': truncated_mean,
#                            'Interquartile Mean': interquartile_mean,
#                            'Midrange': midrange,
#                            'Midhinge': midhinge,
#                            'Trimean': trimean,
#                            'Mean of 20th, 50th, and 80th percentiles': percentiles_mean,
#                            'Winsorized Mean': winsorized_mean,
#                            '1st Quartile': first_quartile,
#                            'Median': median,
#                            '3rd Quartile': third_quartile}
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
# def bootstrap_measures_variance(data_frame, variable_names, n_bootstrap_samples=1000):
#     statistics = pd.DataFrame()
#
#     for variable_name in variable_names:
#         data = data_frame[variable_name]
#         bs_samples = bootstrap_samples(data, n_bootstrap_samples)
#
#         # Calculate minimum
#         min_vals  = [np.min(bs_sample) for bs_sample in bs_samples]
#         min_val =  np.mean(min_vals)
#
#         # Calculate maximum
#         max_vals  = [np.max(bs_sample) for bs_sample in bs_samples]
#         max_val =  np.mean(max_vals)
#
#         # Calculate variance
#         variances = [np.var(bs_sample) for bs_sample in bs_samples]
#         variance = np.mean(variances)
#
#         # Calculate standard deviation
#         std_devs = [np.std(bs_sample) for bs_sample in bs_samples]
#         std_dev = np.mean(std_devs)
#
#         # Calculate range
#         ranges = [np.ptp(bs_sample) for bs_sample in bs_samples]
#         rng = np.mean(ranges)
#
#         # Calculate interquartile range
#         iqrs = [iqr(bs_sample) for bs_sample in bs_samples]
#         iqr_val = np.mean(iqrs)
#
#         # Calculate skewness
#         skewnesses = [skew(bs_sample) for bs_sample in bs_samples]
#         skewness = np.mean(skewnesses)
#
#         # Calculate kurtosis
#         kurtoses = [kurtosis(bs_sample) for bs_sample in bs_samples]
#         kurt = np.mean(kurtoses)
#
#         # Calculate Gini coefficient
#         gini_coeffs = [gini(bs_sample) for bs_sample in bs_samples]
#         gini_coeff = np.mean(gini_coeffs)
#
#         # Create a dictionary with the statistics
#         statistics_dict = {'Variable': variable_name,
#                            'Minimum': min_val,
#                            'Maximum': max_val,
#                            'Variance': variance,
#                            'Standard Deviation': std_dev,
#                            'Range': rng,
#                            'Interquartile Range': iqr_val,
#                            'Skewness': skewness,
#                            'Kurtosis': kurt,
#                            'Gini Coefficient': gini_coeff}
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
# variable_names = market.columns.tolist()  # Get the list of all variable names
# # result_ct = bootstrap_central_tendencies(market, variable_names)
# result_mv = bootstrap_measures_variance(market, variable_names)
# # print(result_ct)
# print(result_mv)
#
# # Save the result to a CSV file
# # result_ct.to_csv('bootstrap_statistics_CT.csv', index=False)
# result_mv.to_csv('bootstrap_statistics_MV.csv', index=False)

# Load the datasets into pandas DataFrames
object = pd.read_csv('object.csv')
object = object[['rooms', 'square_total', 'square_living', 'square_kitchen',
                'ratio_liv_tot', 'ratio_kit_tot', 'floor', 'floors', 'age']]

# Get the list of variables from the datasets
variables = object.columns.tolist()

def percentile_of_value(data_frame, values_df, variable_names, n_bootstrap_samples=1000):
    percentiles_df = pd.DataFrame()

    for variable_name in variable_names:
        data = data_frame[variable_name]
        bs_samples = bootstrap_samples(data, n_bootstrap_samples)
        value = values_df[variable_name].iloc[0]  # Assuming each variable has a single value in values_df

        # Flatten the bootstrap samples to get a single list of all sample values
        all_bs_samples = [item for sublist in bs_samples for item in sublist]

        # Calculate the percentile of the value relative to the bootstrap samples
        percentile = percentileofscore(all_bs_samples, value) / 100

        # Create a dictionary with the variable name and percentile
        percentile_dict = {'Variable': variable_name, 'Percentile': percentile}

        # Create a data frame from the dictionary
        percentile_df = pd.DataFrame(percentile_dict, index=[0])

        # Concatenate the percentile data frame with the existing 'percentiles_df' data frame
        percentiles_df = pd.concat([percentiles_df, percentile_df], ignore_index=True)

    return percentiles_df

values_df = object  # Replace with your specific values
result_percentiles = percentile_of_value(market, values_df, variables)
print(result_percentiles)
result_percentiles.to_csv('bootstrapped_percentiles.csv', index=False)
