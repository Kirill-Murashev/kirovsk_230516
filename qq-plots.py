import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Read data
file_path = 'kirovsk_230516.csv'
market = pd.read_csv(file_path)
market = market[['exposition_days', 'total_discount', 'price', 'unit_price', 'square_total',
                 'square_living', 'square_kitchen', 'ratio_liv_tot', 'ratio_kit_tot', 'floor', 'floors', 'age']]

###############
# Exposure period
# Create variables
value_1 = 'exposition_days'
first_plot = 'raw values'
second_plot = 'logarithms of the values'
third_plot = 'square roots of the values'
fourth_plot = 'Box-Cox transformation of the values'

# Calculate the desirable values
df = pd.DataFrame()
df[f'{value_1}'] = market[f'{value_1}']
df[f'log_{value_1}'] = np.log(market[f'{value_1}'])
df[f'sqrt_{value_1}'] = np.sqrt(market[f'{value_1}'])
bcx_target, lam = stats.boxcox(market[f'{value_1}'])
df[f'bcx_{value_1}'] = bcx_target

# Check the intermediate result
print(df)

# Create a 2x2 subplot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Create a Q-Q plot
for ax, data, name in zip([ax1, ax2, ax3, ax4], [df[f'{value_1}'],
                                                 df[f'log_{value_1}'],
                                                 df[f'sqrt_{value_1}'],
                                                 df[f'bcx_{value_1}']],
                          [f"{value_1}: {first_plot}",
                           f"log_{value_1}: {second_plot}",
                           f"sqrt_{value_1}: {third_plot}",
                           f"bcx_{value_1}: {fourth_plot}"]):
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist='norm', fit=True)
    ax.plot(osm, osr, marker='.', linestyle='none')
    ax.plot(osm, slope*osm + intercept, 'r', label='r={:.2f}'.format(r))
    ax.legend()
    ax.set_title('Q-Q plot for {}'.format(name))

# Save plot to file
plt.tight_layout()
plt.savefig(f'qq-plot_{value_1}.png', dpi=300, bbox_inches='tight')

###############
# Total discount
# Create variables
value_1 = 'total_discount'
first_plot = 'raw values'
second_plot = 'logarithms of the values'
third_plot = 'square roots of the values'
fourth_plot = 'Box-Cox transformation of the values'

# Calculate the desirable values
df = pd.DataFrame()
df[f'{value_1}'] = market[f'{value_1}']
df[f'log_{value_1}'] = np.log(market[f'{value_1}'])
df[f'sqrt_{value_1}'] = np.sqrt(market[f'{value_1}'])
bcx_target, lam = stats.boxcox(market[f'{value_1}'])
df[f'bcx_{value_1}'] = bcx_target

# Check the intermediate result
print(df)

# Create a 2x2 subplot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Create a Q-Q plot
for ax, data, name in zip([ax1, ax2, ax3, ax4], [df[f'{value_1}'],
                                                 df[f'log_{value_1}'],
                                                 df[f'sqrt_{value_1}'],
                                                 df[f'bcx_{value_1}']],
                          [f"{value_1}: {first_plot}",
                           f"log_{value_1}: {second_plot}",
                           f"sqrt_{value_1}: {third_plot}",
                           f"bcx_{value_1}: {fourth_plot}"]):
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist='norm', fit=True)
    ax.plot(osm, osr, marker='.', linestyle='none')
    ax.plot(osm, slope*osm + intercept, 'r', label='r={:.2f}'.format(r))
    ax.legend()
    ax.set_title('Q-Q plot for {}'.format(name))

# Save plot to file
plt.tight_layout()
plt.savefig(f'qq-plot_{value_1}.png', dpi=300, bbox_inches='tight')

###############
# Price
# Create variables
value_1 = 'price'
first_plot = 'raw values'
second_plot = 'logarithms of the values'
third_plot = 'square roots of the values'
fourth_plot = 'Box-Cox transformation of the values'

# Calculate the desirable values
df = pd.DataFrame()
df[f'{value_1}'] = market[f'{value_1}']
df[f'log_{value_1}'] = np.log(market[f'{value_1}'])
df[f'sqrt_{value_1}'] = np.sqrt(market[f'{value_1}'])
bcx_target, lam = stats.boxcox(market[f'{value_1}'])
df[f'bcx_{value_1}'] = bcx_target

# Check the intermediate result
print(df)

# Create a 2x2 subplot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Create a Q-Q plot
for ax, data, name in zip([ax1, ax2, ax3, ax4], [df[f'{value_1}'],
                                                 df[f'log_{value_1}'],
                                                 df[f'sqrt_{value_1}'],
                                                 df[f'bcx_{value_1}']],
                          [f"{value_1}: {first_plot}",
                           f"log_{value_1}: {second_plot}",
                           f"sqrt_{value_1}: {third_plot}",
                           f"bcx_{value_1}: {fourth_plot}"]):
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist='norm', fit=True)
    ax.plot(osm, osr, marker='.', linestyle='none')
    ax.plot(osm, slope*osm + intercept, 'r', label='r={:.2f}'.format(r))
    ax.legend()
    ax.set_title('Q-Q plot for {}'.format(name))

# Save plot to file
plt.tight_layout()
plt.savefig(f'qq-plot_{value_1}.png', dpi=300, bbox_inches='tight')

###############
# Unit price
# Create variables
value_1 = 'unit_price'
first_plot = 'raw values'
second_plot = 'logarithms of the values'
third_plot = 'square roots of the values'
fourth_plot = 'Box-Cox transformation of the values'

# Calculate the desirable values
df = pd.DataFrame()
df[f'{value_1}'] = market[f'{value_1}']
df[f'log_{value_1}'] = np.log(market[f'{value_1}'])
df[f'sqrt_{value_1}'] = np.sqrt(market[f'{value_1}'])
bcx_target, lam = stats.boxcox(market[f'{value_1}'])
df[f'bcx_{value_1}'] = bcx_target

# Check the intermediate result
print(df)

# Create a 2x2 subplot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Create a Q-Q plot
for ax, data, name in zip([ax1, ax2, ax3, ax4], [df[f'{value_1}'],
                                                 df[f'log_{value_1}'],
                                                 df[f'sqrt_{value_1}'],
                                                 df[f'bcx_{value_1}']],
                          [f"{value_1}: {first_plot}",
                           f"log_{value_1}: {second_plot}",
                           f"sqrt_{value_1}: {third_plot}",
                           f"bcx_{value_1}: {fourth_plot}"]):
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist='norm', fit=True)
    ax.plot(osm, osr, marker='.', linestyle='none')
    ax.plot(osm, slope*osm + intercept, 'r', label='r={:.2f}'.format(r))
    ax.legend()
    ax.set_title('Q-Q plot for {}'.format(name))

# Save plot to file
plt.tight_layout()
plt.savefig(f'qq-plot_{value_1}.png', dpi=300, bbox_inches='tight')

###############
# Total square
# Create variables
value_1 = 'square_total'
first_plot = 'raw values'
second_plot = 'logarithms of the values'
third_plot = 'square roots of the values'
fourth_plot = 'Box-Cox transformation of the values'

# Calculate the desirable values
df = pd.DataFrame()
df[f'{value_1}'] = market[f'{value_1}']
df[f'log_{value_1}'] = np.log(market[f'{value_1}'])
df[f'sqrt_{value_1}'] = np.sqrt(market[f'{value_1}'])
bcx_target, lam = stats.boxcox(market[f'{value_1}'])
df[f'bcx_{value_1}'] = bcx_target

# Check the intermediate result
print(df)

# Create a 2x2 subplot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Create a Q-Q plot
for ax, data, name in zip([ax1, ax2, ax3, ax4], [df[f'{value_1}'],
                                                 df[f'log_{value_1}'],
                                                 df[f'sqrt_{value_1}'],
                                                 df[f'bcx_{value_1}']],
                          [f"{value_1}: {first_plot}",
                           f"log_{value_1}: {second_plot}",
                           f"sqrt_{value_1}: {third_plot}",
                           f"bcx_{value_1}: {fourth_plot}"]):
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist='norm', fit=True)
    ax.plot(osm, osr, marker='.', linestyle='none')
    ax.plot(osm, slope*osm + intercept, 'r', label='r={:.2f}'.format(r))
    ax.legend()
    ax.set_title('Q-Q plot for {}'.format(name))

# Save plot to file
plt.tight_layout()
plt.savefig(f'qq-plot_{value_1}.png', dpi=300, bbox_inches='tight')

###############
# Living square
# Create variables
value_1 = 'square_living'
first_plot = 'raw values'
second_plot = 'logarithms of the values'
third_plot = 'square roots of the values'
fourth_plot = 'Box-Cox transformation of the values'

# Calculate the desirable values
df = pd.DataFrame()
df[f'{value_1}'] = market[f'{value_1}']
df[f'log_{value_1}'] = np.log(market[f'{value_1}'])
df[f'sqrt_{value_1}'] = np.sqrt(market[f'{value_1}'])
bcx_target, lam = stats.boxcox(market[f'{value_1}'])
df[f'bcx_{value_1}'] = bcx_target

# Check the intermediate result
print(df)

# Create a 2x2 subplot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Create a Q-Q plot
for ax, data, name in zip([ax1, ax2, ax3, ax4], [df[f'{value_1}'],
                                                 df[f'log_{value_1}'],
                                                 df[f'sqrt_{value_1}'],
                                                 df[f'bcx_{value_1}']],
                          [f"{value_1}: {first_plot}",
                           f"log_{value_1}: {second_plot}",
                           f"sqrt_{value_1}: {third_plot}",
                           f"bcx_{value_1}: {fourth_plot}"]):
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist='norm', fit=True)
    ax.plot(osm, osr, marker='.', linestyle='none')
    ax.plot(osm, slope*osm + intercept, 'r', label='r={:.2f}'.format(r))
    ax.legend()
    ax.set_title('Q-Q plot for {}'.format(name))

# Save plot to file
plt.tight_layout()
plt.savefig(f'qq-plot_{value_1}.png', dpi=300, bbox_inches='tight')

###############
# Kitchens square
# Create variables
value_1 = 'square_kitchen'
first_plot = 'raw values'
second_plot = 'logarithms of the values'
third_plot = 'square roots of the values'
fourth_plot = 'Box-Cox transformation of the values'

# Calculate the desirable values
df = pd.DataFrame()
df[f'{value_1}'] = market[f'{value_1}']
df[f'log_{value_1}'] = np.log(market[f'{value_1}'])
df[f'sqrt_{value_1}'] = np.sqrt(market[f'{value_1}'])
bcx_target, lam = stats.boxcox(market[f'{value_1}'])
df[f'bcx_{value_1}'] = bcx_target

# Check the intermediate result
print(df)

# Create a 2x2 subplot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Create a Q-Q plot
for ax, data, name in zip([ax1, ax2, ax3, ax4], [df[f'{value_1}'],
                                                 df[f'log_{value_1}'],
                                                 df[f'sqrt_{value_1}'],
                                                 df[f'bcx_{value_1}']],
                          [f"{value_1}: {first_plot}",
                           f"log_{value_1}: {second_plot}",
                           f"sqrt_{value_1}: {third_plot}",
                           f"bcx_{value_1}: {fourth_plot}"]):
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist='norm', fit=True)
    ax.plot(osm, osr, marker='.', linestyle='none')
    ax.plot(osm, slope*osm + intercept, 'r', label='r={:.2f}'.format(r))
    ax.legend()
    ax.set_title('Q-Q plot for {}'.format(name))

# Save plot to file
plt.tight_layout()
plt.savefig(f'qq-plot_{value_1}.png', dpi=300, bbox_inches='tight')

###############
# Ratio liv_tot
# Create variables
value_1 = 'ratio_liv_tot'
first_plot = 'raw values'
second_plot = 'logarithms of the values'
third_plot = 'square roots of the values'
fourth_plot = 'Box-Cox transformation of the values'

# Calculate the desirable values
df = pd.DataFrame()
df[f'{value_1}'] = market[f'{value_1}']
df[f'log_{value_1}'] = np.log(market[f'{value_1}'])
df[f'sqrt_{value_1}'] = np.sqrt(market[f'{value_1}'])
bcx_target, lam = stats.boxcox(market[f'{value_1}'])
df[f'bcx_{value_1}'] = bcx_target

# Check the intermediate result
print(df)

# Create a 2x2 subplot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Create a Q-Q plot
for ax, data, name in zip([ax1, ax2, ax3, ax4], [df[f'{value_1}'],
                                                 df[f'log_{value_1}'],
                                                 df[f'sqrt_{value_1}'],
                                                 df[f'bcx_{value_1}']],
                          [f"{value_1}: {first_plot}",
                           f"log_{value_1}: {second_plot}",
                           f"sqrt_{value_1}: {third_plot}",
                           f"bcx_{value_1}: {fourth_plot}"]):
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist='norm', fit=True)
    ax.plot(osm, osr, marker='.', linestyle='none')
    ax.plot(osm, slope*osm + intercept, 'r', label='r={:.2f}'.format(r))
    ax.legend()
    ax.set_title('Q-Q plot for {}'.format(name))

# Save plot to file
plt.tight_layout()
plt.savefig(f'qq-plot_{value_1}.png', dpi=300, bbox_inches='tight')

###############
# Ratio kit_tot
# Create variables
value_1 = 'ratio_kit_tot'
first_plot = 'raw values'
second_plot = 'logarithms of the values'
third_plot = 'square roots of the values'
fourth_plot = 'Box-Cox transformation of the values'

# Calculate the desirable values
df = pd.DataFrame()
df[f'{value_1}'] = market[f'{value_1}']
df[f'log_{value_1}'] = np.log(market[f'{value_1}'])
df[f'sqrt_{value_1}'] = np.sqrt(market[f'{value_1}'])
bcx_target, lam = stats.boxcox(market[f'{value_1}'])
df[f'bcx_{value_1}'] = bcx_target

# Check the intermediate result
print(df)

# Create a 2x2 subplot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Create a Q-Q plot
for ax, data, name in zip([ax1, ax2, ax3, ax4], [df[f'{value_1}'],
                                                 df[f'log_{value_1}'],
                                                 df[f'sqrt_{value_1}'],
                                                 df[f'bcx_{value_1}']],
                          [f"{value_1}: {first_plot}",
                           f"log_{value_1}: {second_plot}",
                           f"sqrt_{value_1}: {third_plot}",
                           f"bcx_{value_1}: {fourth_plot}"]):
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist='norm', fit=True)
    ax.plot(osm, osr, marker='.', linestyle='none')
    ax.plot(osm, slope*osm + intercept, 'r', label='r={:.2f}'.format(r))
    ax.legend()
    ax.set_title('Q-Q plot for {}'.format(name))

# Save plot to file
plt.tight_layout()
plt.savefig(f'qq-plot_{value_1}.png', dpi=300, bbox_inches='tight')

###############
# Floor
# Create variables
value_1 = 'floor'
first_plot = 'raw values'
second_plot = 'logarithms of the values'
third_plot = 'square roots of the values'
fourth_plot = 'Box-Cox transformation of the values'

# Calculate the desirable values
df = pd.DataFrame()
df[f'{value_1}'] = market[f'{value_1}']
df[f'log_{value_1}'] = np.log(market[f'{value_1}'])
df[f'sqrt_{value_1}'] = np.sqrt(market[f'{value_1}'])
bcx_target, lam = stats.boxcox(market[f'{value_1}'])
df[f'bcx_{value_1}'] = bcx_target

# Check the intermediate result
print(df)

# Create a 2x2 subplot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Create a Q-Q plot
for ax, data, name in zip([ax1, ax2, ax3, ax4], [df[f'{value_1}'],
                                                 df[f'log_{value_1}'],
                                                 df[f'sqrt_{value_1}'],
                                                 df[f'bcx_{value_1}']],
                          [f"{value_1}: {first_plot}",
                           f"log_{value_1}: {second_plot}",
                           f"sqrt_{value_1}: {third_plot}",
                           f"bcx_{value_1}: {fourth_plot}"]):
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist='norm', fit=True)
    ax.plot(osm, osr, marker='.', linestyle='none')
    ax.plot(osm, slope*osm + intercept, 'r', label='r={:.2f}'.format(r))
    ax.legend()
    ax.set_title('Q-Q plot for {}'.format(name))

# Save plot to file
plt.tight_layout()
plt.savefig(f'qq-plot_{value_1}.png', dpi=300, bbox_inches='tight')

###############
# Floors
# Create variables
value_1 = 'floors'
first_plot = 'raw values'
second_plot = 'logarithms of the values'
third_plot = 'square roots of the values'
fourth_plot = 'Box-Cox transformation of the values'

# Calculate the desirable values
df = pd.DataFrame()
df[f'{value_1}'] = market[f'{value_1}']
df[f'log_{value_1}'] = np.log(market[f'{value_1}'])
df[f'sqrt_{value_1}'] = np.sqrt(market[f'{value_1}'])
bcx_target, lam = stats.boxcox(market[f'{value_1}'])
df[f'bcx_{value_1}'] = bcx_target

# Check the intermediate result
print(df)

# Create a 2x2 subplot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Create a Q-Q plot
for ax, data, name in zip([ax1, ax2, ax3, ax4], [df[f'{value_1}'],
                                                 df[f'log_{value_1}'],
                                                 df[f'sqrt_{value_1}'],
                                                 df[f'bcx_{value_1}']],
                          [f"{value_1}: {first_plot}",
                           f"log_{value_1}: {second_plot}",
                           f"sqrt_{value_1}: {third_plot}",
                           f"bcx_{value_1}: {fourth_plot}"]):
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist='norm', fit=True)
    ax.plot(osm, osr, marker='.', linestyle='none')
    ax.plot(osm, slope*osm + intercept, 'r', label='r={:.2f}'.format(r))
    ax.legend()
    ax.set_title('Q-Q plot for {}'.format(name))

# Save plot to file
plt.tight_layout()
plt.savefig(f'qq-plot_{value_1}.png', dpi=300, bbox_inches='tight')

###############
# Age
# Create variables
value_1 = 'age'
first_plot = 'raw values'
second_plot = 'logarithms of the values'
third_plot = 'square roots of the values'
fourth_plot = 'Box-Cox transformation of the values'

# Calculate the desirable values
df = pd.DataFrame()
df[f'{value_1}'] = market[f'{value_1}']
df[f'log_{value_1}'] = np.log(market[f'{value_1}'])
df[f'sqrt_{value_1}'] = np.sqrt(market[f'{value_1}'])
bcx_target, lam = stats.boxcox(market[f'{value_1}'])
df[f'bcx_{value_1}'] = bcx_target

# Check the intermediate result
print(df)

# Create a 2x2 subplot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

# Create a Q-Q plot
for ax, data, name in zip([ax1, ax2, ax3, ax4], [df[f'{value_1}'],
                                                 df[f'log_{value_1}'],
                                                 df[f'sqrt_{value_1}'],
                                                 df[f'bcx_{value_1}']],
                          [f"{value_1}: {first_plot}",
                           f"log_{value_1}: {second_plot}",
                           f"sqrt_{value_1}: {third_plot}",
                           f"bcx_{value_1}: {fourth_plot}"]):
    (osm, osr), (slope, intercept, r) = stats.probplot(data, dist='norm', fit=True)
    ax.plot(osm, osr, marker='.', linestyle='none')
    ax.plot(osm, slope*osm + intercept, 'r', label='r={:.2f}'.format(r))
    ax.legend()
    ax.set_title('Q-Q plot for {}'.format(name))

# Save plot to file
plt.tight_layout()
plt.savefig(f'qq-plot_{value_1}.png', dpi=300, bbox_inches='tight')
