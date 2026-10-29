"""Train the used-car price model from Cleaned Car.csv and save it to LinearRegressionModel.pkl.

Usage: python train.py
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = 'Cleaned Car.csv'
MODEL_PATH = 'LinearRegressionModel.pkl'
CATEGORICAL = ['name', 'company', 'fuel_type']
FEATURES = ['name', 'company', 'year', 'kms_driven', 'fuel_type']
SEED = 42


def load_data(path=DATA_PATH):
    car = pd.read_csv(path, index_col=0)
    return car[FEATURES], car['Price']


def build_pipeline():
    # Unseen car names at prediction time are encoded as all zeros instead of raising an error
    column_trans = make_column_transformer(
        (OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
        remainder='passthrough')
    return make_pipeline(column_trans, LinearRegression())


def train(path=DATA_PATH, model_path=MODEL_PATH, verbose=True):
    X, y = load_data(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    cv_scores = cross_val_score(build_pipeline(), X, y, scoring='r2',
                                cv=KFold(n_splits=5, shuffle=True, random_state=SEED))
    if verbose:
        print(f'Test R^2: {r2_score(y_test, y_pred):.3f}')
        print(f'Test MAE: Rs {mean_absolute_error(y_test, y_pred):,.0f}')
        print(f'5-fold CV R^2: {np.mean(cv_scores):.3f} +/- {np.std(cv_scores):.3f}')

    # Refit on all the data for the deployed model
    pipe = build_pipeline().fit(X, y)
    with open(model_path, 'wb') as f:
        pickle.dump(pipe, f)
    if verbose:
        print(f'Saved model to {model_path}')
    return pipe


if __name__ == '__main__':
    train()
