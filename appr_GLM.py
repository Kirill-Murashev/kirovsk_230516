import statsmodels.api as sm
from sklearn.feature_selection import f_regression
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np



def appr_glm_regression_stepwise(data, dependent_var, independent_vars, family):
    # Perform forward selection
    included_vars = []
    while len(independent_vars) > 0:
        best_var = None
        best_pvalue = float('inf')
        for var in independent_vars:
            formula = f"{dependent_var} ~ {' + '.join(included_vars + [var])}"
            model = sm.GLM.from_formula(formula=formula, data=data, family=family)
            results = model.fit()
            pvalue = results.pvalues[var]
            if pvalue < best_pvalue:
                best_var = var
                best_pvalue = pvalue
        if best_var is not None:
            included_vars.append(best_var)
            independent_vars.remove(best_var)
        else:
            break

    # Fit the final model
    formula = f"{dependent_var} ~ {' + '.join(included_vars)}"
    model = sm.GLM.from_formula(formula=formula, data=data, family=family)
    results = model.fit()

    # Print the summary statistics
    print(results.summary())

        # Return the results object
    return results, included_vars


def appr_plot_glm_diagnostics(results):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.suptitle('Diagnostic Plots')

    # Residual plot
    plt.subplot(2, 2, 1)
    residuals = results.resid_deviance
    plt.scatter(results.fittedvalues, residuals)
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')

    # Q-Q plot
    plt.subplot(2, 2, 2)
    sm.graphics.qqplot(results.resid_deviance, dist=stats.norm, line='45', fit=True, ax=plt.gca())
    plt.title('Q-Q Plot')

    # Scale-location plot
    plt.subplot(2, 2, 3)
    sqrt_abs_residuals = np.sqrt(np.abs(results.resid_pearson))
    plt.scatter(results.fittedvalues, sqrt_abs_residuals)
    plt.xlabel('Fitted Values')
    plt.ylabel('sqrt(|Standardized Residuals|)')
    plt.title('Scale-Location Plot')

    # Deviance residuals plot
    # Predicted vs. Actual plot
    plt.subplot(2, 2, 4)
    predicted_values = results.predict()
    actual_values = results.model.endog
    plt.scatter(predicted_values, actual_values)
    plt.plot(actual_values, actual_values, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Predicted vs. Actual Plot')

    plt.tight_layout()
    plt.savefig('/home/kaarlahti/PycharmProjects/kirovsk_230516/regression/GLM.png',
                dpi=300, bbox_inches='tight')
    plt.show()

