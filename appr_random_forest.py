import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


def train_random_forest(df, target_column, params, test_size=0.2, random_state=42):
    # Prepare the features (X) and the target (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split the dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Perform grid search for hyperparameter tuning
    grid = GridSearchCV(RandomForestRegressor(), params, refit=True, verbose=3)

    # Start time
    start_time = time.time()

    grid.fit(X_train, y_train)

    # End time
    end_time = time.time()

    # Print out the best parameters
    print("Best parameters found: ", grid.best_params_)

    # Train the model with the best parameters
    model = grid.best_estimator_

    # Predict on the test set
    predictions = model.predict(X_test)

    # Calculate and print the mean squared error
    mse = mean_squared_error(y_test, predictions)
    print("MSE: ", mse)

    # Calculate and print the execution time
    print("Execution time: ", end_time - start_time)

    # Plot feature importance
    feature_importance = model.feature_importances_
    sns.barplot(x=feature_importance, y=X.columns)
    plt.title('Feature Importance')
    plt.savefig(f'/home/kaarlahti/PycharmProjects/kirovsk_230516/img/random_forest.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    return model


def predict_new_data_rf(model, new_df):
    return model.predict(new_df)


# Define the parameters for the grid search
params = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
}
