import time
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def train_xgboost(df, target_column, params, test_size=0.2, random_state=42):
    # Prepare the features (X) and the target (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split the dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Perform grid search for hyperparameter tuning
    start_time = time.time()
    grid = GridSearchCV(XGBRegressor(use_label_encoder=False), params, verbose=3)
    grid.fit(X_train, y_train)
    end_time = time.time()
    print("Training time: ", end_time - start_time)

    # Print out the best parameters
    print("Best parameters found: ", grid.best_params_)

    # Train the model with the best parameters
    model = grid.best_estimator_

    # Predict on the test set
    predictions = model.predict(X_test)

    # Calculate and print the mean squared error
    mse = mean_squared_error(y_test, predictions)
    print("MSE: ", mse)

    # Plot feature importances
    plt.bar(X.columns, model.feature_importances_)
    plt.xlabel('Features')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.savefig(f'/home/kaarlahti/PycharmProjects/kirovsk_230516/img/xgboost.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    return model


def predict_new_data(model, new_df):
    return model.predict(new_df)


# Define the parameters for the grid search
params = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 500]
}