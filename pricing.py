from pathlib import Path

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
import numpy as np


FEATURES = ['Area', 'Room', 'Parking', 'Warehouse', 'Elevator', 'Address']
DATA = Path(__file__).with_name('housePrice.csv')


def load_data():
    frame = pd.read_csv(DATA, usecols=FEATURES + ['Price'])
    frame['Area'] = pd.to_numeric(frame['Area'], errors='coerce')
    frame['Price'] = pd.to_numeric(frame['Price'], errors='coerce')
    frame = frame.dropna().drop_duplicates()
    return frame[frame['Area'].between(10, 2000) & frame['Price'].gt(0)]


def fit_estimator(features, target):
    transform = ColumnTransformer([
        ('neighborhood', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Address']),
    ], remainder='passthrough').set_output(transform='pandas')
    model = make_pipeline(transform, LGBMRegressor(n_estimators=300, n_jobs=2, random_state=42, verbosity=-1))
    estimator = TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
    return estimator.fit(features, target)


def main():
    frame = load_data()
    train, test = train_test_split(frame, test_size=.2, random_state=42)
    model = fit_estimator(train[FEATURES], train['Price'])
    estimate = model.predict(test[FEATURES])
    baseline = np.full(len(test), train['Price'].median())
    for name, predicted in [('training median', baseline), ('LightGBM', estimate)]:
        print(f"{name:20} MAE {mean_absolute_error(test['Price'], predicted):,.0f} | "
              f"median absolute error {median_absolute_error(test['Price'], predicted):,.0f}")


if __name__ == '__main__':
    main()
