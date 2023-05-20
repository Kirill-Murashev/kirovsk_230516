# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Read and transform data
file = 'sberindex_deals.csv'
df = pd.read_csv(file)
df = df[['date', 'price']]

# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Set 'date' column as index and sort data by index
df.set_index('date', inplace=True)
df = df.sort_index()

# Print the first few rows of the dataframe
print(df.head())

# Plot the diagram
# plt.figure(figsize=(10, 6))
# plt.plot(df.index, df['price'])
# plt.title('Price over time')
# plt.xlabel('Date')
# plt.ylabel('Price')
# plt.show()


# Define function for the test of stationary
def test_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])


# Check if price is stationary
test_stationarity(df['price'])

# Differencing the series
df['price_diff'] = df['price'] - df['price'].shift(1)

# Drop NA values
df.dropna(inplace=True)

# Check if price_diff is stationary
test_stationarity(df['price_diff'])

# Differencing the series
df['price_diff_1'] = df['price_diff'] - df['price_diff'].shift(1)

# Drop NA values
df.dropna(inplace=True)

# Check if price_diff is stationary
test_stationarity(df['price_diff_1'])

# Differencing the series
df['price_diff_2'] = df['price_diff_1'] - df['price_diff_1'].shift(1)

# Drop NA values
df.dropna(inplace=True)

# Check if price_diff is stationary
test_stationarity(df['price_diff_2'])

# Fit auto_arima function
model = auto_arima(df['price_diff_2'],
                   start_p=0, start_q=0,  # initial guess for p and q
                   test='adf',  # use adftest to find optimal 'd'
                   max_p=3, max_q=3,  # maximum p and q
                   m=1,  # frequency of series
                   d=None,  # let model determine 'd'
                   seasonal=False,  # No Seasonality
                   start_P=0,
                   D=0,
                   trace=True,
                   error_action='ignore',
                   suppress_warnings=True,
                   stepwise=True)

# Print model summary
print(model.summary())

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

# Plot the ACF on ax1
plot_acf(df['price_diff_2'], lags=20, zero=False, ax=ax1)

# Plot the PACF on ax2
plot_pacf(df['price_diff_2'], lags=20, zero=False, ax=ax2)

plt.show()

# Get the string representation of the summary
# summary_string = str(model.summary())

# Write the summary string to a text file
# with open('auto_ARIMA_summary.txt', 'w') as file:
#     file.write(summary_string)


model_1 = ARIMA(df['price'], order=(2, 3, 0))
model_fit = model_1.fit()

# Summary of the model
print(model_fit.summary())

# Forecast
n_periods = 12  # for example, forecast for next 12 months
forecast_object = model_fit.get_forecast(steps=n_periods)

# You can obtain the forecast, standard error and confidence intervals as follows:
fc = forecast_object.predicted_mean
conf = forecast_object.conf_int()


# Create a series for plotting
fc_series = pd.Series(fc, index=pd.date_range(start=df.index[-1], periods=n_periods, freq='M'))
lower_series = pd.Series(conf.iloc[:, 0], index=pd.date_range(start=df.index[-1], periods=n_periods, freq='M'))
upper_series = pd.Series(conf.iloc[:, 1], index=pd.date_range(start=df.index[-1], periods=n_periods, freq='M'))

# Plot
plt.figure(figsize=(10, 5))
plt.plot(df['price'])
plt.plot(fc_series, color='red')
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
plt.title("Final Forecast")
plt.show()

# Create a DataFrame
forecast_df = pd.DataFrame({
    'Date': fc_series.index,
    'Forecast': fc_series.values,
    'Lower_Bound': lower_series.values,
    'Upper_Bound': upper_series.values,
})

# Save to CSV
# forecast_df.to_csv('forecast.csv', index=False)

