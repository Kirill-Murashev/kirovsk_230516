from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np


def appr_train_best_ridge_regression(data, target_variable, test_size=0.2, random_state=42):
    """
    Train a Ridge Regression model on the provided dataset.

    Parameters:
    data: The complete dataset.
    target_variable: The dependent variable.
    test_size: Proportion of the dataset to include in the test split.
    random_state: Controls the shuffling applied to the data before applying the split.

    Returns:
    ridge_cv: Trained Ridge Regression model with the best alpha.
    metrics: A dictionary of model metrics (R-squared, RMSE, MAE, BIC, AIC, coefficients).
    """
    # Separate the features and the target variable
    X = data.drop(target_variable, axis=1)
    y = data[target_variable]

    # Split the dataset into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define the alpha values to test
    alphas = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]

    # Train the Ridge Regression model
    ridge_cv = RidgeCV(alphas=alphas)
    ridge_cv.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = ridge_cv.predict(X_test)

    # Calculate metrics
    r2_score = ridge_cv.score(X_test, y_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    n = X_test.shape[0]  # number of observations
    p = X_test.shape[1]  # number of predictors

    # Computing AIC and BIC
    resid = y_test - y_pred
    sse = sum(resid ** 2)
    loglikelihood = -n / 2 * np.log(2 * np.pi * sse / n) - n / 2
    aic = 2 * p - 2 * loglikelihood
    bic = np.log(n) * p - 2 * loglikelihood

    # Coefficients
    coefficients = pd.DataFrame()
    coefficients['Variables'] = X.columns
    coefficients['Coefficients'] = ridge_cv.coef_

    print(f"Best alpha: {ridge_cv.alpha_}")
    print(f"R2-score: {r2_score}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Akaike Information Criterion (AIC): {aic}")
    print(f"Bayesian Information Criterion (BIC): {bic}")
    print("Coefficients:")
    print(coefficients)

    metrics = {'r2_score': r2_score, 'rmse': rmse, 'mae': mae, 'aic': aic, 'bic': bic, 'coefficients': coefficients}

    # Return a function for making predictions with the trained model
    def predict(new_data):
        # Ensure new_data is a DataFrame
        if not isinstance(new_data, pd.DataFrame):
            new_data = pd.DataFrame(new_data)

        # Align the columns of the new data with the training data
        new_data = new_data.reindex(columns=X.columns, fill_value=0)

        # Preprocess the new data
        new_data_processed = scaler.transform(new_data)

        # Make predictions
        predictions = ridge_cv.predict(new_data_processed)

        return predictions

    return ridge_cv, metrics, predict, X.columns

