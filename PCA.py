import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from correlation import calculate_correlations
from correlation import plot_correlations
from correlation import plot_correlation_matrix

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


# Add new columns
market['epoch'] = market['age'].apply(apply_conditions)


# Visualize the rational number of components
def perform_pca_and_plot(df, cols, output_file):
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[cols])

    # Perform PCA on the standardized data
    pca = PCA().fit(df_scaled)

    # Plot the explained variance ratio
    plt.figure(figsize=(12, 8))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.title('Choice the rational number of the principal components based on the Rule of 95% variance')

    # Save the plot to the specified output file
    plt.savefig(output_file, format='png', dpi=300)
    plt.close()


def perform_pca(df, cols, n_components):
    # Standardizing the features
    x = StandardScaler().fit_transform(df[cols])

    # Performing PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(x)

    # Convert principal components to pandas data frame
    principal_df = pd.DataFrame(data=principal_components,
                                columns=["square_PC_" + str(i) for i in range(1, n_components + 1)])

    # Concatenate principal components dataframe to the original dataframe
    final_df = pd.concat([df, principal_df], axis=1)

    return final_df


# Apply function to data for understanding of rational number of the components
# perform_pca_and_plot(market, ['square_total', 'square_living', 'square_kitchen', 'ratio_liv_tot', 'ratio_kit_tot'],
#                       '/home/kaarlahti/PycharmProjects/kirovsk_230516/img/pca_number.png')

market = perform_pca(market, ['square_total', 'square_living', 'square_kitchen', 'ratio_liv_tot', 'ratio_kit_tot'], 2)
print(market)

def pca_plot(df, cols, plot_type='scree', filename='pca_plot.png'):
    # Standardize the data
    data = StandardScaler().fit_transform(df[cols])

    # Perform PCA
    pca = PCA().fit(data)
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Create the desired plot
    plt.figure(figsize=(12, 8))

    if plot_type == 'scree':
        plt.bar(range(len(explained_variance)), explained_variance, alpha=0.5, align='center',
                label='Individual explained variance')
        plt.step(range(len(cumulative_variance)), cumulative_variance, where='mid',
                 label='Cumulative explained variance')
        plt.axhline(y=0.95, color='r', linestyle='-', label="95% variance threshold")
        plt.xlabel('Principal Component')
        plt.ylabel('Variance (%)')
        plt.title('Explained Variance Per Principal Component')
        plt.legend(loc='best')

    elif plot_type == 'cumulative':
        plt.plot(range(len(cumulative_variance)), cumulative_variance)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance as a Function of the Number of Components')


    elif plot_type == 'biplot':

        pca_transformed = PCA(n_components=2).fit_transform(data)

        plt.scatter(pca_transformed[:, 0], pca_transformed[:, 1])

        # Draw the loading vectors

        for i, var_name in enumerate(cols):
            plt.arrow(0, 0, pca.components_[0, i] * max(pca_transformed[:, 0]),
                      pca.components_[1, i] * max(pca_transformed[:, 1]), color='r',
                      alpha=0.5, width=0.005, head_width=0.03, label='Variable vectors')
            plt.text(pca.components_[0, i] * max(pca_transformed[:, 0]),
                     pca.components_[1, i] * max(pca_transformed[:, 1]), var_name, color='black',
                     ha='center', va='center', fontsize=8)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('PCA Biplot')
        plt.grid()

    # Save the plot
    plt.savefig(filename, format='png', dpi=300)

    # Display the plot
    plt.show()


def perform_pca_new_df(df, cols, n_components=None):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    data_scaled = scaler.fit_transform(df[cols])
    pca_result = pca.fit_transform(data_scaled)

    for i in range(pca_result.shape[1]):
        df['PC' + str(i + 1)] = pca_result[:, i]

    return df, pca, scaler


# pca_plot(market, ['square_total', 'square_living', 'square_kitchen', 'ratio_liv_tot', 'ratio_kit_tot'],
#                   plot_type='biplot', filename='/home/kaarlahti/PycharmProjects/kirovsk_230516/img/pca_biplot.png')

# # Apply function to data
# market_PC_cor = calculate_correlations(market, 'unit_price', ['square_PC_1', 'square_PC_2'])
# market_PC_cor = market_PC_cor.applymap(lambda x: f'{x:.5f}' if isinstance(x, float) else x)
#
# # Return the result
# print(market_PC_cor)
# market_PC_cor.to_csv('/home/kaarlahti/PycharmProjects/kirovsk_230516/tables/correlations_PC.csv')

# plot_correlations(market, 'unit_price', ['square_PC_1', 'square_PC_2'], 'kendall', 2,
#                   '/home/kaarlahti/PycharmProjects/kirovsk_230516/img/')

# plot_correlation_matrix(market, 'kendall', ['unit_price', 'square_PC_1', 'square_PC_2'],
#                         '/home/kaarlahti/PycharmProjects/kirovsk_230516/img/')

# market_1, pca, scaler = perform_pca_new_df(market, cols=['square_total', 'square_living',
#                                                          'square_kitchen', 'ratio_liv_tot', 'ratio_kit_tot'],
#                                            n_components=2)
#
# object = pd.read_csv('object.csv')
# object = object[['square_total', 'square_living', 'square_kitchen', 'ratio_liv_tot', 'ratio_kit_tot']]
# object = object.values
# object = np.array(object).reshape(1, -1)
#
# # Standardize the new data
# object_standardized = scaler.transform(object)
# # Transform the new data to the PCA space
# new_data_pca = pca.transform(object_standardized)
# print(new_data_pca)

print(market)

market.to_csv('market.csv')