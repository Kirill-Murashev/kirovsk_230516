# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Read the CSV file and create a DataFrame object
csv_file_path = 'kirovsk_230516.csv'
kir_market_2305 = pd.read_csv(csv_file_path, sep=';', header=0)

print(kir_market_2305.head())

# Create new dataframe with only two variables: exposition_days, discount_0
df_discount = kir_market_2305[['exposition_days', 'discount_0']]
print(df_discount.head())

df_discount = df_discount.sort_values(by='exposition_days')
print(df_discount.head())

df_discount['exposition_days'] = np.log(df_discount['exposition_days'])
print(df_discount.head())

df_discount['discount_0'] = df_discount['discount_0'].ewm(alpha=0.2).mean()
print(df_discount.head(5))

df_discount = df_discount.rename(columns={'exposition_days': 'exposure_days_log', 'discount_0': 'smoothed_discount'})
print(df_discount.head(5))

plt.figure(figsize=(10, 6))
plt.scatter(df_discount['exposure_days_log'], df_discount['smoothed_discount'], s=50, c='blue', marker='o')
plt.xlabel('Exposure days (log)', fontsize=14)
plt.ylabel('Discount coefficient', fontsize=14)
plt.title('Scatter plot: Days vs. Discount', fontsize=16)
plt.grid(True)
plt.savefig('scatter_days_vs_discount.png')
plt.draw()
plt.pause(10)

#
# Set the plot style and size
sns.set(style='whitegrid', font_scale=1.5)

# Draw a scatter plot with a linear trend line and confidence intervals
lm = sns.lmplot(x='exposure_days_log', y='smoothed_discount', data=df_discount,
           scatter_kws={'s': 50},    # Increase the size of scatter points
           line_kws={'color': 'red'}  # Change the color of the trend line to red
           )

plt.title('Scatter plot with trend line: Days vs. Discount')
# Adjust the size of the figure
lm.fig.set_size_inches(8.3, 11.7)  # A4 dimensions
plt.show()


# Add a constant to the independent variable
X = sm.add_constant(df_discount['exposure_days_log'])

# Define the dependent variable
Y = df_discount['smoothed_discount']

# Fit the model
model = sm.OLS(Y, X)
results = model.fit()

# Print the summary statistics of the regression model
print(results.summary())

kir_market_2305['expected_discount'] = 1.0321 - 0.0178 * df_discount['exposure_days_log']

print(kir_market_2305['expected_discount'])