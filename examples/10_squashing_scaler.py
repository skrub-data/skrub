import warnings

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skrub import DatetimeEncoder, SquashingScaler, TableVectorizer
from skrub.datasets import fetch_employee_salaries

for num_transformer in [StandardScaler(), SquashingScaler()]:
    np.random.seed(0)
    data = fetch_employee_salaries()

    model = TransformedTargetRegressor(MLPRegressor(), transformer=StandardScaler())
    scoring = "r2"

    pipeline = Pipeline(
        steps=[
            (
                "tv",
                TableVectorizer(datetime=DatetimeEncoder(periodic_encoding="circular")),
            ),
            ("num", num_transformer),
            ("model", model),
        ]
    )

    with warnings.catch_warnings():
        # ignore warning about unknown categories
        warnings.simplefilter("ignore", category=UserWarning)

        scores = cross_val_score(pipeline, data.X, data.y, cv=3, scoring=scoring)

    print(
        f"Cross-validation R2 scores for {num_transformer.__class__.__name__} (higher"
        f" is better): {scores}"
    )
