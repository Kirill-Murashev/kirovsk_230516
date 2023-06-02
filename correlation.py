# Import libraries
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os

# Set pandas display options
pd.set_option('display.float_format', lambda x: '%.5f' % x)

# Read data
file_path = 'kirovsk_230516.csv'
market = pd.read_csv(file_path)


# # Create function for correlation calculating
# def calculate_correlations(df, dependent_var, independent_vars, alpha = 0.05):
#     result = []
#     for independent_var in independent_vars:
#         pearson_coef, pearson_p = stats.pearsonr(df[dependent_var], df[independent_var])
#         spearman_coef, spearman_p = stats.spearmanr(df[dependent_var], df[independent_var])
#         kendall_coef, kendall_p = stats.kendalltau(df[dependent_var], df[independent_var])
#
#         result.append({
#             'independent_var': independent_var,
#             'pearson_coef': pearson_coef,
#             'pearson_p': pearson_p,
#             'pearson_significant': 'yes' if pearson_p < alpha else 'no',
#             'spearman_coef': spearman_coef,
#             'spearman_p': spearman_p,
#             'spearman_significant': 'yes' if spearman_p < alpha else 'no',
#             'kendall_coef': kendall_coef,
#             'kendall_p': kendall_p,
#             'kendall_significant': 'yes' if kendall_p < alpha else 'no'
#         })
#
#     result_df = pd.DataFrame(result)
#     result_df.set_index('independent_var', inplace=True)
#
#     return result_df
#
# # Apply function to data
# market_cor = calculate_correlations(market, 'unit_price', ['square_total', 'square_living', 'square_kitchen',
#                                                            'ratio_liv_tot', 'ratio_kit_tot', 'age'])
# market_cor = market_cor.applymap(lambda x: f'{x:.5f}' if isinstance(x, float) else x)
#
# # Return the result
# print(market_cor)
# market_cor.to_csv('/home/kaarlahti/PycharmProjects/kirovsk_230516/tables/correlations.csv')

# # calculate correlation matrix
# corr_matrix = market[['square_total', 'square_living', 'square_kitchen', 'ratio_liv_tot',
#               'ratio_kit_tot', 'age'] + ['unit_price']].corr()
#
# # create a mask for the upper triangle
# mask = np.triu(np.ones_like(corr, dtype=bool))
#
# # set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))
#
# # generate a custom diverging colormap
# cmap = sns.diverging_palette(230, 20, as_cmap=True)
#
# # draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)
#
# print(corr_matrix)
# plt.show()
# Save plot to file


# def plot_correlations(df, dependent_var, independent_vars, correlation_type='pearson', plots_per_row=3, path='.'):
#     if correlation_type == 'pearson':
#         correlation_func = stats.pearsonr
#     elif correlation_type == 'spearman':
#         correlation_func = stats.spearmanr
#     elif correlation_type == 'kendall':
#         correlation_func = stats.kendalltau
#     else:
#         raise ValueError("Invalid correlation_type. Choose 'pearson', 'spearman', or 'kendall'.")
#
#     num_plots = len(independent_vars)
#     num_rows = math.ceil(num_plots / plots_per_row)
#
#     fig, axs = plt.subplots(num_rows, plots_per_row, figsize=(15, num_rows * 5))
#     axs = axs.flatten()
#
#     for ax, independent_var in zip(axs, independent_vars):
#         sns.scatterplot(data=df, x=independent_var, y=dependent_var, ax=ax)
#
#         coef, p = correlation_func(df[dependent_var], df[independent_var])
#         ax.set_title(f'{correlation_type.capitalize()} Correlation: {coef:.2f}, p-value: {p:.2f}')
#
#     # remove empty subplots
#     for i in range(num_plots, len(axs)):
#         fig.delaxes(axs[i])
#
#     plt.tight_layout()
#
#     # save the plot to a .png file
#     plt.savefig(os.path.join(path, f'{correlation_type}_{dependent_var}_correlation.png'))
#
#     plt.show()
#
#
# plot_correlations(market, 'unit_price', ['square_total', 'square_living',
#                                          'square_kitchen', 'ratio_liv_tot', 'ratio_kit_tot', 'age'],
#                   'kendall', 2, '/home/kaarlahti/PycharmProjects/kirovsk_230516/img/')


def plot_correlation_matrix(df, correlation_type, vars, path):
    # Check if correlation type is valid
    if correlation_type not in ['pearson', 'spearman', 'kendall']:
        print(f'Invalid correlation type: {correlation_type}')
        return

    # Check if specified variables exist in the dataframe
    for var in vars:
        if var not in df.columns:
            print(f'Variable not found in dataframe: {var}')
            return

    # Check if file path is valid
    if not os.path.isdir(path):
        print(f'Invalid file path: {path}')
        return

    # Select correlation function based on the correlation type
    corr_func = {
        'pearson': lambda x, y: stats.pearsonr(x, y)[0],
        'spearman': lambda x, y: stats.spearmanr(x, y)[0],
        'kendall': lambda x, y: stats.kendalltau(x, y)[0]
    }[correlation_type]

    p_func = {
        'pearson': lambda x, y: stats.pearsonr(x, y)[1],
        'spearman': lambda x, y: stats.spearmanr(x, y)[1],
        'kendall': lambda x, y: stats.kendalltau(x, y)[1]
    }[correlation_type]

    # Subset dataframe and compute correlation and p-value matrices
    df = df[vars]
    corr_matrix = df.corr(method=corr_func)
    p_matrix = df.corr(method=p_func)

    # Combine the correlation and p-value in one cell with precision of 5 decimal places
    combined_matrix = corr_matrix.applymap("{0:.5f}".format) + "\n(" + p_matrix.applymap("{0:.5f}".format) + ")"

    # Create mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Plot the correlation matrix
    plt.figure(figsize=(12, 8))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, annot=combined_matrix, fmt='',
                annot_kws={'size': 10}, linewidths=.5, cbar_kws={"shrink": .5},
                xticklabels=1, yticklabels=1)  # Decrease label size
    plt.title(f'{correlation_type.capitalize()} Correlation Matrix', fontsize=16)

    # Save the plot to a .png file
    file_path = os.path.join(path, f'{correlation_type}_correlation_matrix.png')
    plt.savefig(file_path, dpi=300)
    print(f'Saved plot to: {file_path}')

    # Show the plot
    plt.show()


plot_correlation_matrix(market, 'kendall', ['unit_price', 'square_total', 'square_living',
                                            'square_kitchen', 'ratio_liv_tot', 'ratio_kit_tot', 'age'],
                        '/home/kaarlahti/PycharmProjects/kirovsk_230516/img/')
