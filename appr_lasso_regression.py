from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def appr_train_best_lasso_regression(data, target_variable, test_size=0.2, random_state=42):
    X = data.drop(target_variable, axis=1)
    y = data[target_variable]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LassoCV(cv=5, random_state=random_state).fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    metrics = calculate_metrics(y_train, y_test, y_train_pred, y_test_pred, model, X_train)

    def predict(new_data):
        new_data = new_data.reindex(columns=X.columns, fill_value=0)
        new_data_scaled = scaler.transform(new_data)
        return model.predict(new_data_scaled)

    return model, metrics, predict, X_train.columns


def calculate_metrics(y_train, y_test, y_train_pred, y_test_pred, model, X_train):
    metrics = {'Train': {
        'MSE': mean_squared_error(y_train, y_train_pred),
        'MAE': mean_absolute_error(y_train, y_train_pred),
        'R2': r2_score(y_train, y_train_pred),
        'Adjusted R2': 1 - (1 - r2_score(y_train, y_train_pred)) * (len(y_train) - 1) / (
                len(y_train) - X_train.shape[1] - 1)
    }, 'Test': {
        'MSE': mean_squared_error(y_test, y_test_pred),
        'MAE': mean_absolute_error(y_test, y_test_pred),
        'R2': r2_score(y_test, y_test_pred),
        'Adjusted R2': 1 - (1 - r2_score(y_test, y_test_pred)) * (len(y_test) - 1) / (
                len(y_test) - X_train.shape[1] - 1)
    }}

    metrics['Train']['BIC'] = calculate_bic(metrics['Train']['MSE'], len(y_train), X_train.shape[1])
    metrics['Test']['BIC'] = calculate_bic(metrics['Test']['MSE'], len(y_test), X_train.shape[1])
    metrics['Train']['AIC'] = calculate_aic(metrics['Train']['MSE'], len(y_train), X_train.shape[1])
    metrics['Test']['AIC'] = calculate_aic(metrics['Test']['MSE'], len(y_test), X_train.shape[1])

    metrics['Coefficients'] = model.coef_

    return metrics


def calculate_bic(mse, n_samples, n_features):
    return n_samples * np.log(mse) + n_features * np.log(n_samples)


def calculate_aic(mse, n_samples, n_features):
    return n_samples * np.log(mse) + 2 * n_features
