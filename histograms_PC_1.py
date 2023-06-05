import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox, norm
import seaborn as sns

# Read data
file_path = 'market.csv'
market = pd.read_csv(file_path)

###############
# PC_1
# Calculate the desired variables

# Create variables
value_1 = 'square_PC_1'
value_2 = 'square_PC_1'

df_PC1 = pd.DataFrame()
df_PC1[f'{value_1}'] = market[f'{value_1}']

df_PC1[f'log_{value_1}'] = np.log(market[f'{value_1}'] + 5)
df_PC1[f'sqrt_{value_1}'] = np.sqrt(market[f'{value_1}'] + 5)
bcx_target, lam = boxcox(market[f'{value_1}'] + 5)
df_PC1[f'bcx_{value_1}'] = bcx_target

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Raw data
sns.histplot(df_PC1[f'{value_1}'], kde=True, stat="density",
             label=f'{value_2}', ax=axs[0, 0])
axs[0, 0].set_xlabel(f'Raw {value_1}')
axs[0, 0].set_ylabel('Density')
axs[0, 0].set_title(f'Histogram of {value_2} (raw values)')

# Add normal curve to raw data histogram
mu_raw, std_raw = np.mean(df_PC1[f'{value_1}']),\
    np.std(df_PC1[f'{value_1}'])
x_raw = np.linspace(mu_raw - 3 * std_raw, mu_raw + 3 * std_raw, 100)
axs[0, 0].plot(x_raw, norm.pdf(x_raw, mu_raw, std_raw),
               color='red', label='Normal curve')
axs[0, 0].legend()

# Log
sns.histplot(df_PC1[f'log_{value_1}'], kde=True, stat="density",
             label=f'Logarithm of {value_2}', ax=axs[0, 1])
axs[0, 1].set_xlabel(f'Logarithm of {value_1}')
axs[0, 1].set_ylabel('Density')
axs[0, 1].set_title(f'Histogram of {value_2} (logarithms of values)')

# Add normal curve to log histogram
mu_log, std_log = np.mean(df_PC1[f'log_{value_1}']),\
    np.std(df_PC1[f'log_{value_1}'])
x_log = np.linspace(mu_log - 3 * std_log, mu_log + 3 * std_log, 100)
axs[0, 1].plot(x_log, norm.pdf(x_log, mu_log, std_log),
               color='red', label='Normal curve')
axs[0, 1].legend()

# Square root
sns.histplot(df_PC1[f'sqrt_{value_1}'], kde=True, stat="density",
             label=f'Square root of {value_2}', ax=axs[1, 0])
axs[1, 0].set_xlabel(f'Square root of {value_1}')
axs[1, 0].set_ylabel('Density')
axs[1, 0].set_title(f'Histogram of {value_2} (square roots of values)')

# Add normal curve to square root histogram
mu_sqrt, std_sqrt = np.mean(df_PC1[f'sqrt_{value_1}']),\
    np.std(df_PC1[f'sqrt_{value_1}'])
x_sqrt = np.linspace(mu_sqrt - 3 * std_sqrt, mu_sqrt + 3 * std_sqrt, 100)
axs[1, 0].plot(x_sqrt, norm.pdf(x_sqrt, mu_sqrt, std_sqrt),
               color='red', label='Normal curve')
axs[1, 0].legend()

# Box-Cox
sns.histplot(df_PC1[f'bcx_{value_1}'], kde=True, stat="density",
             label=f'Transformed {value_2}', ax=axs[1, 1])
axs[1, 1].set_xlabel(f'{value_1} with Box-Cox transformation')
axs[1, 1].set_ylabel('Density')
axs[1, 1].set_title(f'Histogram of {value_2} (Box-Cox transformation, lambda = {round(lam, 5)})')

# Add normal curve to Box-Cox histogram
mu_bcx, std_bcx = np.mean(df_PC1[f'bcx_{value_1}']),\
    np.std(df_PC1[f'bcx_{value_1}'])
x_bcx = np.linspace(mu_bcx - 3 * std_bcx, mu_bcx + 3 * std_bcx, 100)
axs[1, 1].plot(x_bcx, norm.pdf(x_bcx, mu_bcx, std_bcx),
               color='red', label='Normal curve')
axs[1, 1].legend()

# Adjust layout
plt.tight_layout()

# Save and display the plot
plt.savefig(f'/home/kaarlahti/PycharmProjects/kirovsk_230516/img/histograms_with_normal_curves_{value_1}.png',
            dpi=300, bbox_inches='tight')
plt.show()

# Release RAM
df_PC1 = None