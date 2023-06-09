import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import time
import matplotlib.pyplot as plt


def train_gradient_boosting(df, target_column, params, test_size=0.2, random_state=42):
    start_time = time.time()

    # Prepare the features (X) and the target (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split the dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Perform grid search for hyperparameter tuning
    grid = GridSearchCV(GradientBoostingRegressor(), params, refit=True, verbose=3)
    grid.fit(X_train, y_train)

    # Print out the best parameters
    print("Best parameters found: ", grid.best_params_)

    # Train the model with the best parameters
    model = grid.best_estimator_

    # Predict on the test set
    predictions = model.predict(X_test)

    # Calculate and print the mean squared error
    mse = mean_squared_error(y_test, predictions)
    print("MSE: ", mse)

    # Plot feature importance
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    fig = plt.figure(figsize=(12, 8))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, np.array(df.columns)[sorted_idx])
    plt.title('Feature Importance (MDI)')
    plt.savefig(f'/home/kaarlahti/PycharmProjects/kirovsk_230516/img/gradient_boosting.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    end_time = time.time()
    print(f"Time taken to train Gradient Boosting model and make predictions: {end_time - start_time} seconds")

    return model


# Define the parameters for the grid search
params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.1, 0.05, 0.01],
    'max_depth': [4, 6, 8],
    'min_samples_leaf': [20, 50, 100, 150],
}
