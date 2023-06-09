import pandas as pd
import numpy as np
import pymc3 as pm
import theano.tensor as tt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Prepare training and test datasets
boston = datasets.load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.Series(boston.target, name='MEDV')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
def train_bayesian_regression(X, y):
    with pm.Model() as model:
        # Prior distributions for the model parameters
        alpha = pm.Normal('alpha', mu=0, sd=10)
        betas = pm.Normal('betas', mu=0, sd=10, shape=X.shape[1])
        sigma = pm.HalfNormal('sigma', sd=1)

        # Expected value of outcome
        mu = alpha + tt.dot(X, betas)

        # Likelihood (sampling distribution) of observations
        Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=y)

        # Compute trace
        trace = pm.sample(2000, tune=1000)

    return model, trace

model, trace = train_bayesian_regression(X_train, y_train)

# Model summary
pm.summary(trace).round(2)

# Prediction
def predict(model, trace, X_new):
    with model:
        pm.set_data({
            'X': X_new
        })
        post_pred = pm.sample_posterior_predictive(trace, samples=100)
        y_pred = np.mean(post_pred['Y_obs'], axis=0)

    return y_pred

y_pred = predict(model, trace, X_test)
