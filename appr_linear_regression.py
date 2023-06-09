# Import libraries
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression


def appr_multiple_regression(data: pd.DataFrame, dependent_var: str):
    independent_vars = data.columns.tolist()
    independent_vars.remove(dependent_var)

    X = data[independent_vars]
    Y = data[dependent_var]

    # Adding constant to the independent variables
    X = sm.add_constant(X)

    model = sm.OLS(Y, X)
    results = model.fit()

    return results


def appr_multiple_regression_auto(data: pd.DataFrame, dependent_var: str):
    independent_vars = data.columns.tolist()
    independent_vars.remove(dependent_var)

    p_value_max = 1
    while len(independent_vars) > 0 and p_value_max > 0.05:
        X = data[independent_vars]
        X = sm.add_constant(X)
        model = sm.OLS(data[dependent_var], X)
        results = model.fit()
        p_values = results.pvalues
        p_value_max = max(p_values)
        if p_value_max > 0.05:
            excluded_variable = p_values.idxmax()
            independent_vars.remove(excluded_variable)

    return results, independent_vars


def appr_multiple_regression_auto_cv(data: pd.DataFrame, dependent_var: str):
    independent_vars = data.columns.tolist()
    independent_vars.remove(dependent_var)

    p_value_max = 1
    while len(independent_vars) > 0 and p_value_max > 0.05:
        X = data[independent_vars]
        Y = data[dependent_var]

        model = LinearRegression()
        scores = cross_val_score(model, X, Y, cv=5)

        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        results = model.fit()
        p_values = results.pvalues
        p_value_max = max(p_values)
        if p_value_max > 0.05:
            excluded_variable = p_values.idxmax()
            independent_vars.remove(excluded_variable)

    print('Cross-validated scores:', scores)
    return results, independent_vars


def appr_plot_residuals(results):
    plt.figure(figsize=(12, 8))
    residuals = results.resid
    fitted = results.fittedvalues
    sns.residplot(x=fitted, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plt.title('Residuals vs Fitted')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.savefig('/home/kaarlahti/PycharmProjects/kirovsk_230516/regression/linear_regression_residuals.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def create_summary(results):
    # First, create a DataFrame with the statistics for each parameter
    params_summary = pd.DataFrame({
        'coef': results.params,
        'std err': results.bse,
        't-stat': results.tvalues,
        'p-value': results.pvalues,
        'Conf. Int. Low': results.conf_int()[0],
        'Conf. Int. Up.': results.conf_int()[1],
    })

    # Next, create a Series with the overall model statistics
    model_summary = pd.Series({
        'R-squared': results.rsquared,
        'Adj. R-squared': results.rsquared_adj,
        'F-statistic': results.fvalue,
        'Prob (F-statistic)': results.f_pvalue,
        'AIC': results.aic,
        'BIC': results.bic,
    })

    # Round all values in the DataFrame and Series to 5 decimal places
    params_summary = params_summary.round(5)
    model_summary = model_summary.round(5)

    return model_summary, params_summary


def preprocess_data(data: pd.DataFrame, dependent_var: str, categorical_vars: list):
    # Create list of independent variables
    independent_vars = [var for var in data.columns if var != dependent_var]

    # Convert categorical variables to dummy variables
    data = pd.get_dummies(data, columns=categorical_vars, drop_first=True)

    # Cast all dummies to float
    for column in data.columns:
        if column not in independent_vars and column != dependent_var:
            data[column] = data[column].astype(float)

    # Update independent_vars after get_dummies
    independent_vars = [var for var in data.columns if var != dependent_var]

    return data, independent_vars


def align_new_data(new_data: pd.DataFrame, training_vars: list) -> pd.DataFrame:
    missing_cols = set(training_vars) - set(new_data.columns)
    for c in missing_cols:
        new_data[c] = 0
    extra_cols = set(new_data.columns) - set(training_vars)
    if extra_cols:
        print(f"Warning: new data has columns not found in training data: {extra_cols}")
        new_data = new_data.drop(columns=extra_cols)
    return new_data[training_vars]


def predict_new_data(new_data, model_results, model_vars):
    new_data = align_new_data(new_data, model_vars)
    new_data = sm.add_constant(new_data)
    predictions = model_results.predict(new_data)
    return predictions

import matplotlib.pyplot as plt
import pandas as pd


def visualize_regression_coefficients(results):
    # Coefficient plots
    plt.figure(figsize=(12, 8))
    coef = pd.DataFrame(results.params)
    coef.columns = ['coef']
    coef.sort_values(by='coef', inplace=True)
    coef['coef'].plot(kind='barh', color='blue')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Coefficient Name')
    plt.title('Coefficient plot')
    plt.tight_layout()
    plt.savefig('/home/kaarlahti/PycharmProjects/kirovsk_230516/regression/linear_regression_coefficients.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def visualize_scatter_predicted_vs_actual(results, data, dependent_var):
    plt.figure(figsize=(12, 8))
    plt.scatter(data[dependent_var], results.predict())
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.title('Actual vs Predicted values')
    plt.plot([min(data[dependent_var]), max(data[dependent_var])],
             [min(data[dependent_var]), max(data[dependent_var])],
             color='red')  # Line of perfect fit
    plt.savefig('/home/kaarlahti/PycharmProjects/kirovsk_230516/regression/linear_regression_actual_vs_predicted.png',
                dpi=300, bbox_inches='tight')
    plt.show()
