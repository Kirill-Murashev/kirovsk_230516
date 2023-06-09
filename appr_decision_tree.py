from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.metrics import mean_squared_error
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

def train_decision_tree(df, target_column, params, test_size=0.2, random_state=42):
    # Prepare the features (X) and the target (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Split the dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Perform grid search for hyperparameter tuning
    grid = GridSearchCV(DecisionTreeRegressor(), params, refit=True, verbose=3)
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

    plt.figure(figsize=(32, 16))  # Set the figure size (optional)
    plot_tree(grid.best_estimator_, filled=True, feature_names=X.columns, fontsize=10)
    plt.title('The decision Tree', fontsize=20)
    plt.savefig(f'/home/kaarlahti/PycharmProjects/kirovsk_230516/img/decision_tree.png',
                dpi=300, bbox_inches='tight')
    plt.show()

    return model


def predict_new_data(model, new_df):
    return model.predict(new_df)


# Define the parameters for the grid search
params = {
    'max_depth': [10, 20, 30, None],
    'min_samples_leaf': [1, 2, 4],
    'min_samples_split': [2, 5, 10],
}

# Assume we have a DataFrame `df` for training and `new_df` for prediction
# model = train_decision_tree(df, 'target', params)
# predictions = predict_new_data(model, new_df)
