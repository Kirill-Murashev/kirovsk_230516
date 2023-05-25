import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox, norm
import seaborn as sns

# Read data
file_path = 'kirovsk_230516.csv'
df_kir = pd.read_csv(file_path)

###############
# Price
# Calculate the desired variables

# Create variables
value_1 = 'price'
value_2 = 'prices'

df_price = pd.DataFrame()
df_price[f'{value_1}'] = df_kir[f'{value_1}']

df_price[f'log_{value_1}'] = np.log(df_kir[f'{value_1}'])
df_price[f'sqrt_{value_1}'] = np.sqrt(df_kir[f'{value_1}'])
bcx_target, lam = boxcox(df_kir[f'{value_1}'])
df_price[f'bcx_{value_1}'] = bcx_target

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Raw data
sns.histplot(df_price[f'{value_1}'], kde=True, stat="density",
             label=f'{value_2}', ax=axs[0, 0])
axs[0, 0].set_xlabel(f'Raw {value_1}')
axs[0, 0].set_ylabel('Density')
axs[0, 0].set_title(f'Histogram of {value_2} (raw values)')

# Add normal curve to raw data histogram
mu_raw, std_raw = np.mean(df_price[f'{value_1}']),\
    np.std(df_price[f'{value_1}'])
x_raw = np.linspace(mu_raw - 3 * std_raw, mu_raw + 3 * std_raw, 100)
axs[0, 0].plot(x_raw, norm.pdf(x_raw, mu_raw, std_raw),
               color='red', label='Normal curve')
axs[0, 0].legend()

# Log
sns.histplot(df_price[f'log_{value_1}'], kde=True, stat="density",
             label=f'Logarithm of {value_2}', ax=axs[0, 1])
axs[0, 1].set_xlabel(f'Logarithm of {value_1}')
axs[0, 1].set_ylabel('Density')
axs[0, 1].set_title(f'Histogram of {value_2} (logarithms of values)')

# Add normal curve to log histogram
mu_log, std_log = np.mean(df_price[f'log_{value_1}']),\
    np.std(df_price[f'log_{value_1}'])
x_log = np.linspace(mu_log - 3 * std_log, mu_log + 3 * std_log, 100)
axs[0, 1].plot(x_log, norm.pdf(x_log, mu_log, std_log),
               color='red', label='Normal curve')
axs[0, 1].legend()

# Square root
sns.histplot(df_price[f'sqrt_{value_1}'], kde=True, stat="density",
             label=f'Square root of {value_2}', ax=axs[1, 0])
axs[1, 0].set_xlabel(f'Square root of {value_1}')
axs[1, 0].set_ylabel('Density')
axs[1, 0].set_title(f'Histogram of {value_2} (square roots of values)')

# Add normal curve to square root histogram
mu_sqrt, std_sqrt = np.mean(df_price[f'sqrt_{value_1}']),\
    np.std(df_price[f'sqrt_{value_1}'])
x_sqrt = np.linspace(mu_sqrt - 3 * std_sqrt, mu_sqrt + 3 * std_sqrt, 100)
axs[1, 0].plot(x_sqrt, norm.pdf(x_sqrt, mu_sqrt, std_sqrt),
               color='red', label='Normal curve')
axs[1, 0].legend()

# Box-Cox
sns.histplot(df_price[f'bcx_{value_1}'], kde=True, stat="density",
             label=f'Transformed {value_2}', ax=axs[1, 1])
axs[1, 1].set_xlabel(f'{value_1} with Box-Cox transformation')
axs[1, 1].set_ylabel('Density')
axs[1, 1].set_title(f'Histogram of {value_2} (Box-Cox transformation, lambda = {round(lam, 5)})')

# Add normal curve to Box-Cox histogram
mu_bcx, std_bcx = np.mean(df_price[f'bcx_{value_1}']),\
    np.std(df_price[f'bcx_{value_1}'])
x_bcx = np.linspace(mu_bcx - 3 * std_bcx, mu_bcx + 3 * std_bcx, 100)
axs[1, 1].plot(x_bcx, norm.pdf(x_bcx, mu_bcx, std_bcx),
               color='red', label='Normal curve')
axs[1, 1].legend()

# Adjust layout
plt.tight_layout()

# Save and display the plot
plt.savefig(f'histograms_with_normal_curves_{value_1}.png', dpi=300, bbox_inches='tight')
plt.show()

# Release RAM
df_price = None

###############
# Unit price
# Calculate the desired variables

# Create variables
value_1 = 'unit_price'
value_2 = 'unit_prices'

df_unit_price = pd.DataFrame()
df_unit_price[f'{value_1}'] = df_kir[f'{value_1}']

df_unit_price[f'log_{value_1}'] = np.log(df_kir[f'{value_1}'])
df_unit_price[f'sqrt_{value_1}'] = np.sqrt(df_kir[f'{value_1}'])
bcx_target, lam = boxcox(df_kir[f'{value_1}'])
df_unit_price[f'bcx_{value_1}'] = bcx_target

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Raw data
sns.histplot(df_unit_price[f'{value_1}'], kde=True, stat="density",
             label=f'{value_2}', ax=axs[0, 0])
axs[0, 0].set_xlabel(f'Raw {value_1}')
axs[0, 0].set_ylabel('Density')
axs[0, 0].set_title(f'Histogram of {value_2} (raw values)')

# Add normal curve to raw data histogram
mu_raw, std_raw = np.mean(df_unit_price[f'{value_1}']),\
    np.std(df_unit_price[f'{value_1}'])
x_raw = np.linspace(mu_raw - 3 * std_raw, mu_raw + 3 * std_raw, 100)
axs[0, 0].plot(x_raw, norm.pdf(x_raw, mu_raw, std_raw),
               color='red', label='Normal curve')
axs[0, 0].legend()

# Log
sns.histplot(df_unit_price[f'log_{value_1}'], kde=True, stat="density",
             label=f'Logarithm of {value_2}', ax=axs[0, 1])
axs[0, 1].set_xlabel(f'Logarithm of {value_1}')
axs[0, 1].set_ylabel('Density')
axs[0, 1].set_title(f'Histogram of {value_2} (logarithms of values)')

# Add normal curve to log histogram
mu_log, std_log = np.mean(df_unit_price[f'log_{value_1}']),\
    np.std(df_unit_price[f'log_{value_1}'])
x_log = np.linspace(mu_log - 3 * std_log, mu_log + 3 * std_log, 100)
axs[0, 1].plot(x_log, norm.pdf(x_log, mu_log, std_log),
               color='red', label='Normal curve')
axs[0, 1].legend()

# Square root
sns.histplot(df_unit_price[f'sqrt_{value_1}'], kde=True, stat="density",
             label=f'Square root of {value_2}', ax=axs[1, 0])
axs[1, 0].set_xlabel(f'Square root of {value_1}')
axs[1, 0].set_ylabel('Density')
axs[1, 0].set_title(f'Histogram of {value_2} (square roots of values)')

# Add normal curve to square root histogram
mu_sqrt, std_sqrt = np.mean(df_unit_price[f'sqrt_{value_1}']),\
    np.std(df_unit_price[f'sqrt_{value_1}'])
x_sqrt = np.linspace(mu_sqrt - 3 * std_sqrt, mu_sqrt + 3 * std_sqrt, 100)
axs[1, 0].plot(x_sqrt, norm.pdf(x_sqrt, mu_sqrt, std_sqrt),
               color='red', label='Normal curve')
axs[1, 0].legend()

# Box-Cox
sns.histplot(df_unit_price[f'bcx_{value_1}'], kde=True, stat="density",
             label=f'Transformed {value_2}', ax=axs[1, 1])
axs[1, 1].set_xlabel(f'{value_1} with Box-Cox transformation')
axs[1, 1].set_ylabel('Density')
axs[1, 1].set_title(f'Histogram of {value_2} (Box-Cox transformation, lambda = {round(lam, 5)})')

# Add normal curve to Box-Cox histogram
mu_bcx, std_bcx = np.mean(df_unit_price[f'bcx_{value_1}']),\
    np.std(df_unit_price[f'bcx_{value_1}'])
x_bcx = np.linspace(mu_bcx - 3 * std_bcx, mu_bcx + 3 * std_bcx, 100)
axs[1, 1].plot(x_bcx, norm.pdf(x_bcx, mu_bcx, std_bcx),
               color='red', label='Normal curve')
axs[1, 1].legend()

# Adjust layout
plt.tight_layout()

# Save and display the plot
plt.savefig(f'histograms_with_normal_curves_{value_1}.png', dpi=300, bbox_inches='tight')
plt.show()

# Release RAM
df_unit_price = None

###############
# Square total
# Calculate the desired variables

# Create variables
value_1 = 'square_total'
value_2 = 'squares_total'

df_square_total = pd.DataFrame()
df_square_total[f'{value_1}'] = df_kir[f'{value_1}']

df_square_total[f'log_{value_1}'] = np.log(df_kir[f'{value_1}'])
df_square_total[f'sqrt_{value_1}'] = np.sqrt(df_kir[f'{value_1}'])
bcx_target, lam = boxcox(df_kir[f'{value_1}'])
df_square_total[f'bcx_{value_1}'] = bcx_target

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Raw data
sns.histplot(df_square_total[f'{value_1}'], kde=True, stat="density",
             label=f'{value_2}', ax=axs[0, 0])
axs[0, 0].set_xlabel(f'Raw {value_1}')
axs[0, 0].set_ylabel('Density')
axs[0, 0].set_title(f'Histogram of {value_2} (raw values)')

# Add normal curve to raw data histogram
mu_raw, std_raw = np.mean(df_square_total[f'{value_1}']),\
    np.std(df_square_total[f'{value_1}'])
x_raw = np.linspace(mu_raw - 3 * std_raw, mu_raw + 3 * std_raw, 100)
axs[0, 0].plot(x_raw, norm.pdf(x_raw, mu_raw, std_raw),
               color='red', label='Normal curve')
axs[0, 0].legend()

# Log
sns.histplot(df_square_total[f'log_{value_1}'], kde=True, stat="density",
             label=f'Logarithm of {value_2}', ax=axs[0, 1])
axs[0, 1].set_xlabel(f'Logarithm of {value_1}')
axs[0, 1].set_ylabel('Density')
axs[0, 1].set_title(f'Histogram of {value_2} (logarithms of values)')

# Add normal curve to log histogram
mu_log, std_log = np.mean(df_square_total[f'log_{value_1}']),\
    np.std(df_square_total[f'log_{value_1}'])
x_log = np.linspace(mu_log - 3 * std_log, mu_log + 3 * std_log, 100)
axs[0, 1].plot(x_log, norm.pdf(x_log, mu_log, std_log),
               color='red', label='Normal curve')
axs[0, 1].legend()

# Square root
sns.histplot(df_square_total[f'sqrt_{value_1}'], kde=True, stat="density",
             label=f'Square root of {value_2}', ax=axs[1, 0])
axs[1, 0].set_xlabel(f'Square root of {value_1}')
axs[1, 0].set_ylabel('Density')
axs[1, 0].set_title(f'Histogram of {value_2} (square roots of values)')

# Add normal curve to square root histogram
mu_sqrt, std_sqrt = np.mean(df_square_total[f'sqrt_{value_1}']),\
    np.std(df_square_total[f'sqrt_{value_1}'])
x_sqrt = np.linspace(mu_sqrt - 3 * std_sqrt, mu_sqrt + 3 * std_sqrt, 100)
axs[1, 0].plot(x_sqrt, norm.pdf(x_sqrt, mu_sqrt, std_sqrt),
               color='red', label='Normal curve')
axs[1, 0].legend()

# Box-Cox
sns.histplot(df_square_total[f'bcx_{value_1}'], kde=True, stat="density",
             label=f'Transformed {value_2}', ax=axs[1, 1])
axs[1, 1].set_xlabel(f'{value_1} with Box-Cox transformation')
axs[1, 1].set_ylabel('Density')
axs[1, 1].set_title(f'Histogram of {value_2} (Box-Cox transformation, lambda = {round(lam, 5)})')

# Add normal curve to Box-Cox histogram
mu_bcx, std_bcx = np.mean(df_square_total[f'bcx_{value_1}']),\
    np.std(df_square_total[f'bcx_{value_1}'])
x_bcx = np.linspace(mu_bcx - 3 * std_bcx, mu_bcx + 3 * std_bcx, 100)
axs[1, 1].plot(x_bcx, norm.pdf(x_bcx, mu_bcx, std_bcx),
               color='red', label='Normal curve')
axs[1, 1].legend()

# Adjust layout
plt.tight_layout()

# Save and display the plot
plt.savefig(f'histograms_with_normal_curves_{value_1}.png', dpi=300, bbox_inches='tight')
plt.show()

# Release RAM
df_square_total = None

###############
# Square living
# Calculate the desired variables

# Create variables
value_1 = 'square_living'
value_2 = 'squares_living'

df_square_living = pd.DataFrame()
df_square_living[f'{value_1}'] = df_kir[f'{value_1}']

df_square_living[f'log_{value_1}'] = np.log(df_kir[f'{value_1}'])
df_square_living[f'sqrt_{value_1}'] = np.sqrt(df_kir[f'{value_1}'])
bcx_target, lam = boxcox(df_kir[f'{value_1}'])
df_square_living[f'bcx_{value_1}'] = bcx_target

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Raw data
sns.histplot(df_square_living[f'{value_1}'], kde=True, stat="density",
             label=f'{value_2}', ax=axs[0, 0])
axs[0, 0].set_xlabel(f'Raw {value_1}')
axs[0, 0].set_ylabel('Density')
axs[0, 0].set_title(f'Histogram of {value_2} (raw values)')

# Add normal curve to raw data histogram
mu_raw, std_raw = np.mean(df_square_living[f'{value_1}']),\
    np.std(df_square_living[f'{value_1}'])
x_raw = np.linspace(mu_raw - 3 * std_raw, mu_raw + 3 * std_raw, 100)
axs[0, 0].plot(x_raw, norm.pdf(x_raw, mu_raw, std_raw),
               color='red', label='Normal curve')
axs[0, 0].legend()

# Log
sns.histplot(df_square_living[f'log_{value_1}'], kde=True, stat="density",
             label=f'Logarithm of {value_2}', ax=axs[0, 1])
axs[0, 1].set_xlabel(f'Logarithm of {value_1}')
axs[0, 1].set_ylabel('Density')
axs[0, 1].set_title(f'Histogram of {value_2} (logarithms of values)')

# Add normal curve to log histogram
mu_log, std_log = np.mean(df_square_living[f'log_{value_1}']),\
    np.std(df_square_living[f'log_{value_1}'])
x_log = np.linspace(mu_log - 3 * std_log, mu_log + 3 * std_log, 100)
axs[0, 1].plot(x_log, norm.pdf(x_log, mu_log, std_log),
               color='red', label='Normal curve')
axs[0, 1].legend()

# Square root
sns.histplot(df_square_living[f'sqrt_{value_1}'], kde=True, stat="density",
             label=f'Square root of {value_2}', ax=axs[1, 0])
axs[1, 0].set_xlabel(f'Square root of {value_1}')
axs[1, 0].set_ylabel('Density')
axs[1, 0].set_title(f'Histogram of {value_2} (square roots of values)')

# Add normal curve to square root histogram
mu_sqrt, std_sqrt = np.mean(df_square_living[f'sqrt_{value_1}']),\
    np.std(df_square_living[f'sqrt_{value_1}'])
x_sqrt = np.linspace(mu_sqrt - 3 * std_sqrt, mu_sqrt + 3 * std_sqrt, 100)
axs[1, 0].plot(x_sqrt, norm.pdf(x_sqrt, mu_sqrt, std_sqrt),
               color='red', label='Normal curve')
axs[1, 0].legend()

# Box-Cox
sns.histplot(df_square_living[f'bcx_{value_1}'], kde=True, stat="density",
             label=f'Transformed {value_2}', ax=axs[1, 1])
axs[1, 1].set_xlabel(f'{value_1} with Box-Cox transformation')
axs[1, 1].set_ylabel('Density')
axs[1, 1].set_title(f'Histogram of {value_2} (Box-Cox transformation, lambda = {round(lam, 5)})')

# Add normal curve to Box-Cox histogram
mu_bcx, std_bcx = np.mean(df_square_living[f'bcx_{value_1}']),\
    np.std(df_square_living[f'bcx_{value_1}'])
x_bcx = np.linspace(mu_bcx - 3 * std_bcx, mu_bcx + 3 * std_bcx, 100)
axs[1, 1].plot(x_bcx, norm.pdf(x_bcx, mu_bcx, std_bcx),
               color='red', label='Normal curve')
axs[1, 1].legend()

# Adjust layout
plt.tight_layout()

# Save and display the plot
plt.savefig(f'histograms_with_normal_curves_{value_1}.png', dpi=300, bbox_inches='tight')
plt.show()

# Release RAM
df_square_living = None

###############
# Square kitchen
# Calculate the desired variables

# Create variables
value_1 = 'square_kitchen'
value_2 = 'squares_kitchen'

df_square_kitchen = pd.DataFrame()
df_square_kitchen[f'{value_1}'] = df_kir[f'{value_1}']

df_square_kitchen[f'log_{value_1}'] = np.log(df_kir[f'{value_1}'])
df_square_kitchen[f'sqrt_{value_1}'] = np.sqrt(df_kir[f'{value_1}'])
bcx_target, lam = boxcox(df_kir[f'{value_1}'])
df_square_kitchen[f'bcx_{value_1}'] = bcx_target

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Raw data
sns.histplot(df_square_kitchen[f'{value_1}'], kde=True, stat="density",
             label=f'{value_2}', ax=axs[0, 0])
axs[0, 0].set_xlabel(f'Raw {value_1}')
axs[0, 0].set_ylabel('Density')
axs[0, 0].set_title(f'Histogram of {value_2} (raw values)')

# Add normal curve to raw data histogram
mu_raw, std_raw = np.mean(df_square_kitchen[f'{value_1}']),\
    np.std(df_square_kitchen[f'{value_1}'])
x_raw = np.linspace(mu_raw - 3 * std_raw, mu_raw + 3 * std_raw, 100)
axs[0, 0].plot(x_raw, norm.pdf(x_raw, mu_raw, std_raw),
               color='red', label='Normal curve')
axs[0, 0].legend()

# Log
sns.histplot(df_square_kitchen[f'log_{value_1}'], kde=True, stat="density",
             label=f'Logarithm of {value_2}', ax=axs[0, 1])
axs[0, 1].set_xlabel(f'Logarithm of {value_1}')
axs[0, 1].set_ylabel('Density')
axs[0, 1].set_title(f'Histogram of {value_2} (logarithms of values)')

# Add normal curve to log histogram
mu_log, std_log = np.mean(df_square_kitchen[f'log_{value_1}']),\
    np.std(df_square_kitchen[f'log_{value_1}'])
x_log = np.linspace(mu_log - 3 * std_log, mu_log + 3 * std_log, 100)
axs[0, 1].plot(x_log, norm.pdf(x_log, mu_log, std_log),
               color='red', label='Normal curve')
axs[0, 1].legend()

# Square root
sns.histplot(df_square_kitchen[f'sqrt_{value_1}'], kde=True, stat="density",
             label=f'Square root of {value_2}', ax=axs[1, 0])
axs[1, 0].set_xlabel(f'Square root of {value_1}')
axs[1, 0].set_ylabel('Density')
axs[1, 0].set_title(f'Histogram of {value_2} (square roots of values)')

# Add normal curve to square root histogram
mu_sqrt, std_sqrt = np.mean(df_square_kitchen[f'sqrt_{value_1}']),\
    np.std(df_square_kitchen[f'sqrt_{value_1}'])
x_sqrt = np.linspace(mu_sqrt - 3 * std_sqrt, mu_sqrt + 3 * std_sqrt, 100)
axs[1, 0].plot(x_sqrt, norm.pdf(x_sqrt, mu_sqrt, std_sqrt),
               color='red', label='Normal curve')
axs[1, 0].legend()

# Box-Cox
sns.histplot(df_square_kitchen[f'bcx_{value_1}'], kde=True, stat="density",
             label=f'Transformed {value_2}', ax=axs[1, 1])
axs[1, 1].set_xlabel(f'{value_1} with Box-Cox transformation')
axs[1, 1].set_ylabel('Density')
axs[1, 1].set_title(f'Histogram of {value_2} (Box-Cox transformation, lambda = {round(lam, 5)})')

# Add normal curve to Box-Cox histogram
mu_bcx, std_bcx = np.mean(df_square_kitchen[f'bcx_{value_1}']),\
    np.std(df_square_kitchen[f'bcx_{value_1}'])
x_bcx = np.linspace(mu_bcx - 3 * std_bcx, mu_bcx + 3 * std_bcx, 100)
axs[1, 1].plot(x_bcx, norm.pdf(x_bcx, mu_bcx, std_bcx),
               color='red', label='Normal curve')
axs[1, 1].legend()

# Adjust layout
plt.tight_layout()

# Save and display the plot
plt.savefig(f'histograms_with_normal_curves_{value_1}.png', dpi=300, bbox_inches='tight')
plt.show()

# Release RAM
df_square_kitchen = None

###############
# ratio_liv_tot
# Calculate the desired variables

# Create variables
value_1 = 'ratio_liv_tot'
value_2 = 'ratios_liv_tot'

df_ratio_liv_tot = pd.DataFrame()
df_ratio_liv_tot[f'{value_1}'] = df_kir[f'{value_1}']

df_ratio_liv_tot[f'log_{value_1}'] = np.log(df_kir[f'{value_1}'])
df_ratio_liv_tot[f'sqrt_{value_1}'] = np.sqrt(df_kir[f'{value_1}'])
bcx_target, lam = boxcox(df_kir[f'{value_1}'])
df_ratio_liv_tot[f'bcx_{value_1}'] = bcx_target

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Raw data
sns.histplot(df_ratio_liv_tot[f'{value_1}'], kde=True, stat="density",
             label=f'{value_2}', ax=axs[0, 0])
axs[0, 0].set_xlabel(f'Raw {value_1}')
axs[0, 0].set_ylabel('Density')
axs[0, 0].set_title(f'Histogram of {value_2} (raw values)')

# Add normal curve to raw data histogram
mu_raw, std_raw = np.mean(df_ratio_liv_tot[f'{value_1}']),\
    np.std(df_ratio_liv_tot[f'{value_1}'])
x_raw = np.linspace(mu_raw - 3 * std_raw, mu_raw + 3 * std_raw, 100)
axs[0, 0].plot(x_raw, norm.pdf(x_raw, mu_raw, std_raw),
               color='red', label='Normal curve')
axs[0, 0].legend()

# Log
sns.histplot(df_ratio_liv_tot[f'log_{value_1}'], kde=True, stat="density",
             label=f'Logarithm of {value_2}', ax=axs[0, 1])
axs[0, 1].set_xlabel(f'Logarithm of {value_1}')
axs[0, 1].set_ylabel('Density')
axs[0, 1].set_title(f'Histogram of {value_2} (logarithms of values)')

# Add normal curve to log histogram
mu_log, std_log = np.mean(df_ratio_liv_tot[f'log_{value_1}']),\
    np.std(df_ratio_liv_tot[f'log_{value_1}'])
x_log = np.linspace(mu_log - 3 * std_log, mu_log + 3 * std_log, 100)
axs[0, 1].plot(x_log, norm.pdf(x_log, mu_log, std_log),
               color='red', label='Normal curve')
axs[0, 1].legend()

# Square root
sns.histplot(df_ratio_liv_tot[f'sqrt_{value_1}'], kde=True, stat="density",
             label=f'Square root of {value_2}', ax=axs[1, 0])
axs[1, 0].set_xlabel(f'Square root of {value_1}')
axs[1, 0].set_ylabel('Density')
axs[1, 0].set_title(f'Histogram of {value_2} (square roots of values)')

# Add normal curve to square root histogram
mu_sqrt, std_sqrt = np.mean(df_ratio_liv_tot[f'sqrt_{value_1}']),\
    np.std(df_ratio_liv_tot[f'sqrt_{value_1}'])
x_sqrt = np.linspace(mu_sqrt - 3 * std_sqrt, mu_sqrt + 3 * std_sqrt, 100)
axs[1, 0].plot(x_sqrt, norm.pdf(x_sqrt, mu_sqrt, std_sqrt),
               color='red', label='Normal curve')
axs[1, 0].legend()

# Box-Cox
sns.histplot(df_ratio_liv_tot[f'bcx_{value_1}'], kde=True, stat="density",
             label=f'Transformed {value_2}', ax=axs[1, 1])
axs[1, 1].set_xlabel(f'{value_1} with Box-Cox transformation')
axs[1, 1].set_ylabel('Density')
axs[1, 1].set_title(f'Histogram of {value_2} (Box-Cox transformation, lambda = {round(lam, 5)})')

# Add normal curve to Box-Cox histogram
mu_bcx, std_bcx = np.mean(df_ratio_liv_tot[f'bcx_{value_1}']),\
    np.std(df_ratio_liv_tot[f'bcx_{value_1}'])
x_bcx = np.linspace(mu_bcx - 3 * std_bcx, mu_bcx + 3 * std_bcx, 100)
axs[1, 1].plot(x_bcx, norm.pdf(x_bcx, mu_bcx, std_bcx),
               color='red', label='Normal curve')
axs[1, 1].legend()

# Adjust layout
plt.tight_layout()

# Save and display the plot
plt.savefig(f'histograms_with_normal_curves_{value_1}.png', dpi=300, bbox_inches='tight')
plt.show()

# Release RAM
df_ratio_liv_tot = None

###############
# ratio_kit_tot
# Calculate the desired variables

# Create variables
value_1 = 'ratio_kit_tot'
value_2 = 'ratios_kit_tot'

df_ratio_kit_tot = pd.DataFrame()
df_ratio_kit_tot[f'{value_1}'] = df_kir[f'{value_1}']

df_ratio_kit_tot[f'log_{value_1}'] = np.log(df_kir[f'{value_1}'])
df_ratio_kit_tot[f'sqrt_{value_1}'] = np.sqrt(df_kir[f'{value_1}'])
bcx_target, lam = boxcox(df_kir[f'{value_1}'])
df_ratio_kit_tot[f'bcx_{value_1}'] = bcx_target

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Raw data
sns.histplot(df_ratio_kit_tot[f'{value_1}'], kde=True, stat="density",
             label=f'{value_2}', ax=axs[0, 0])
axs[0, 0].set_xlabel(f'Raw {value_1}')
axs[0, 0].set_ylabel('Density')
axs[0, 0].set_title(f'Histogram of {value_2} (raw values)')

# Add normal curve to raw data histogram
mu_raw, std_raw = np.mean(df_ratio_kit_tot[f'{value_1}']),\
    np.std(df_ratio_kit_tot[f'{value_1}'])
x_raw = np.linspace(mu_raw - 3 * std_raw, mu_raw + 3 * std_raw, 100)
axs[0, 0].plot(x_raw, norm.pdf(x_raw, mu_raw, std_raw),
               color='red', label='Normal curve')
axs[0, 0].legend()

# Log
sns.histplot(df_ratio_kit_tot[f'log_{value_1}'], kde=True, stat="density",
             label=f'Logarithm of {value_2}', ax=axs[0, 1])
axs[0, 1].set_xlabel(f'Logarithm of {value_1}')
axs[0, 1].set_ylabel('Density')
axs[0, 1].set_title(f'Histogram of {value_2} (logarithms of values)')

# Add normal curve to log histogram
mu_log, std_log = np.mean(df_ratio_kit_tot[f'log_{value_1}']),\
    np.std(df_ratio_kit_tot[f'log_{value_1}'])
x_log = np.linspace(mu_log - 3 * std_log, mu_log + 3 * std_log, 100)
axs[0, 1].plot(x_log, norm.pdf(x_log, mu_log, std_log),
               color='red', label='Normal curve')
axs[0, 1].legend()

# Square root
sns.histplot(df_ratio_kit_tot[f'sqrt_{value_1}'], kde=True, stat="density",
             label=f'Square root of {value_2}', ax=axs[1, 0])
axs[1, 0].set_xlabel(f'Square root of {value_1}')
axs[1, 0].set_ylabel('Density')
axs[1, 0].set_title(f'Histogram of {value_2} (square roots of values)')

# Add normal curve to square root histogram
mu_sqrt, std_sqrt = np.mean(df_ratio_kit_tot[f'sqrt_{value_1}']),\
    np.std(df_ratio_kit_tot[f'sqrt_{value_1}'])
x_sqrt = np.linspace(mu_sqrt - 3 * std_sqrt, mu_sqrt + 3 * std_sqrt, 100)
axs[1, 0].plot(x_sqrt, norm.pdf(x_sqrt, mu_sqrt, std_sqrt),
               color='red', label='Normal curve')
axs[1, 0].legend()

# Box-Cox
sns.histplot(df_ratio_kit_tot[f'bcx_{value_1}'], kde=True, stat="density",
             label=f'Transformed {value_2}', ax=axs[1, 1])
axs[1, 1].set_xlabel(f'{value_1} with Box-Cox transformation')
axs[1, 1].set_ylabel('Density')
axs[1, 1].set_title(f'Histogram of {value_2} (Box-Cox transformation, lambda = {round(lam, 5)})')

# Add normal curve to Box-Cox histogram
mu_bcx, std_bcx = np.mean(df_ratio_kit_tot[f'bcx_{value_1}']),\
    np.std(df_ratio_kit_tot[f'bcx_{value_1}'])
x_bcx = np.linspace(mu_bcx - 3 * std_bcx, mu_bcx + 3 * std_bcx, 100)
axs[1, 1].plot(x_bcx, norm.pdf(x_bcx, mu_bcx, std_bcx),
               color='red', label='Normal curve')
axs[1, 1].legend()

# Adjust layout
plt.tight_layout()

# Save and display the plot
plt.savefig(f'histograms_with_normal_curves_{value_1}.png', dpi=300, bbox_inches='tight')
plt.show()

# Release RAM
df_ratio_kit_tot = None

###############
# Age
# Calculate the desired variables
# Create variables
value_1 = 'age'
value_2 = 'age'

df_age = pd.DataFrame()
df_age[f'{value_1}'] = df_kir[f'{value_1}']

df_age[f'log_{value_1}'] = np.log(df_kir[f'{value_1}'])
df_age[f'sqrt_{value_1}'] = np.sqrt(df_kir[f'{value_1}'])
bcx_target, lam = boxcox(df_kir[f'{value_1}'])
df_age[f'bcx_{value_1}'] = bcx_target

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Raw data
sns.histplot(df_age[f'{value_1}'], kde=True, stat="density",
             label=f'{value_2}', ax=axs[0, 0])
axs[0, 0].set_xlabel(f'Raw {value_1}')
axs[0, 0].set_ylabel('Density')
axs[0, 0].set_title(f'Histogram of {value_2} (raw values)')

# Add normal curve to raw data histogram
mu_raw, std_raw = np.mean(df_age[f'{value_1}']),\
    np.std(df_age[f'{value_1}'])
x_raw = np.linspace(mu_raw - 3 * std_raw, mu_raw + 3 * std_raw, 100)
axs[0, 0].plot(x_raw, norm.pdf(x_raw, mu_raw, std_raw),
               color='red', label='Normal curve')
axs[0, 0].legend()

# Log
sns.histplot(df_age[f'log_{value_1}'], kde=True, stat="density",
             label=f'Logarithm of {value_2}', ax=axs[0, 1])
axs[0, 1].set_xlabel(f'Logarithm of {value_1}')
axs[0, 1].set_ylabel('Density')
axs[0, 1].set_title(f'Histogram of {value_2} (logarithms of values)')

# Add normal curve to log histogram
mu_log, std_log = np.mean(df_age[f'log_{value_1}']),\
    np.std(df_age[f'log_{value_1}'])
x_log = np.linspace(mu_log - 3 * std_log, mu_log + 3 * std_log, 100)
axs[0, 1].plot(x_log, norm.pdf(x_log, mu_log, std_log),
               color='red', label='Normal curve')
axs[0, 1].legend()

# Square root
sns.histplot(df_age[f'sqrt_{value_1}'], kde=True, stat="density",
             label=f'Square root of {value_2}', ax=axs[1, 0])
axs[1, 0].set_xlabel(f'Square root of {value_1}')
axs[1, 0].set_ylabel('Density')
axs[1, 0].set_title(f'Histogram of {value_2} (square roots of values)')

# Add normal curve to square root histogram
mu_sqrt, std_sqrt = np.mean(df_age[f'sqrt_{value_1}']),\
    np.std(df_age[f'sqrt_{value_1}'])
x_sqrt = np.linspace(mu_sqrt - 3 * std_sqrt, mu_sqrt + 3 * std_sqrt, 100)
axs[1, 0].plot(x_sqrt, norm.pdf(x_sqrt, mu_sqrt, std_sqrt),
               color='red', label='Normal curve')
axs[1, 0].legend()

# Box-Cox
sns.histplot(df_age[f'bcx_{value_1}'], kde=True, stat="density",
             label=f'Transformed {value_2}', ax=axs[1, 1])
axs[1, 1].set_xlabel(f'{value_1} with Box-Cox transformation')
axs[1, 1].set_ylabel('Density')
axs[1, 1].set_title(f'Histogram of {value_2} (Box-Cox transformation, lambda = {round(lam, 5)})')

# Add normal curve to Box-Cox histogram
mu_bcx, std_bcx = np.mean(df_age[f'bcx_{value_1}']),\
    np.std(df_age[f'bcx_{value_1}'])
x_bcx = np.linspace(mu_bcx - 3 * std_bcx, mu_bcx + 3 * std_bcx, 100)
axs[1, 1].plot(x_bcx, norm.pdf(x_bcx, mu_bcx, std_bcx),
               color='red', label='Normal curve')
axs[1, 1].legend()

# Adjust layout
plt.tight_layout()

# Save and display the plot
plt.savefig(f'histograms_with_normal_curves_{value_1}.png', dpi=300, bbox_inches='tight')
plt.show()

# Release RAM
df_age = None
