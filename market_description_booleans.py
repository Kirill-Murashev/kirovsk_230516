# Import libraries
import pandas as pd

# Read data
file_path = 'kirovsk_230516.csv'
market = pd.read_csv(file_path)
print(market.dtypes)

for col in market.columns:
    if market[col].dropna().isin([0, 1]).all():
        market[col] = market[col].astype(bool)

print(market.dtypes)

# create new DataFrame with only boolean columns
market_bool = market.select_dtypes(include=[bool])
print(market_bool.dtypes)
print(market_bool)

# Release RAM
market = None

# calculate the mean of each column
mean_values = market_bool.mean()

# convert the Series to DataFrame and reset the index
boolean_df = pd.DataFrame(mean_values, columns=['mean']).reset_index()

# rename columns
boolean_df.columns = ['variable', 'mean']

# save the DataFrame to a CSV file
boolean_df.to_csv('mean_booleans.csv', index=False)
