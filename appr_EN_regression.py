import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.eval_measures import bic, aic


def train_best_elastic_net_regression(data, target_column):
    # Define feature matrix and target variable
    X = data.drop(columns=target_column)
    y = data[target_column]

    # Split the data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the Elastic Net regression model
    model = ElasticNetCV(cv=5, random_state=0)

    # Fit the model
    model.fit(X_train_scaled, y_train)

    # Predict
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_adj_r2 = 1 - (1 - train_r2) * (len(y_train) - 1) / (len(y_train) - X_train_scaled.shape[1] - 1)
    test_adj_r2 = 1 - (1 - test_r2) * (len(y_test) - 1) / (len(y_test) - X_test_scaled.shape[1] - 1)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_bic = bic(y_train, y_train_pred, X_train_scaled.shape[1])
    test_bic = bic(y_test, y_test_pred, X_test_scaled.shape[1])

    train_aic = aic(y_train, y_train_pred, X_train_scaled.shape[1])
    test_aic = aic(y_test, y_test_pred, X_test_scaled.shape[1])

    metrics = {
        'Train': [train_r2, train_adj_r2, train_bic, train_aic, train_mse, train_mae],
        'Test': [test_r2, test_adj_r2, test_bic, test_aic, test_mse, test_mae]
    }

    metrics_df = pd.DataFrame(metrics, index=['R2', 'Adj. R2', 'BIC', 'AIC', 'MSE', 'MAE'])

    # Coefficients
    coefficients = pd.DataFrame(np.append(model.intercept_, model.coef_)).transpose()
    coefficients.columns = ['Intercept'] + list(X.columns)

    return {'model': model, 'metrics': metrics_df, 'coefficients': coefficients, 'scaler': scaler}


def predict(model_dict, new_data):
    new_data_scaled = model_dict["scaler"].transform(new_data)  # Scale the new data using the saved scaler
    return model_dict["model"].predict(new_data_scaled)
