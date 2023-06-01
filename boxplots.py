# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

market = None


def plot_violinplots(df, value_col):
    # Get the list of boolean columns
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()

    for i, col in enumerate(bool_cols):
        # Create a new figure for each violin plot
        fig, ax = plt.subplots(figsize=(10, 5))

        # Violin plot
        sns.violinplot(x=col, y=value_col, data=df, ax=ax)

        ax.set_title(f'Violin plot for {col}')

        # Save the plot as a .png file with a unique name
        fig.savefig(f'/home/kaarlahti/PycharmProjects/kirovsk_230516/img/violinplot_{col}_{i}.png')
        plt.close(fig)  # Close the current figure


plot_violinplots(market_bool, 'unit_price')
