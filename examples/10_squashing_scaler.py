import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from skrub import (
    AdaptiveSquashingTransformer,
    DatetimeEncoder,
    TableVectorizer,
)
from skrub.datasets import fetch_employee_salaries


def evaluate(num_transformer):
    data = fetch_employee_salaries()  # works
    # data = fetch_open_payments()
    # data = fetch_medical_charge()  # way too slow, also doesn't work well
    # data = fetch_traffic_violations()  # way too large
    # data = fetch_bike_sharing()  # works
    # data = fetch_drug_directory()
    # X, y = fetch_california_housing(return_X_y=True, as_frame=True)

    # very small, minimal benefits
    # X, y = load_diabetes(return_X_y=True, as_frame=True)
    # data = namedtuple('Bunch', 'X, y')(X, y)
    is_regression = True

    if is_regression:
        # model = MLPRegressor()
        # model = TransformedTargetRegressor(LinearRegression(),
        #   transformer=StandardScaler())
        # model = TransformedTargetRegressor(
        #     MLPRegressor(hidden_layer_sizes=(128, 128), early_stopping=True,
        #     random_state=0, validation_fraction=0.2),
        #     transformer=StandardScaler())
        model = TransformedTargetRegressor(MLPRegressor(), transformer=StandardScaler())
        # model = TransformedTargetRegressor(KernelRidge(kernel='laplacian'),
        #                                    transformer=StandardScaler())
        # model = HistGradientBoostingRegressor()
        scoring = "r2"
    else:
        model = MLPClassifier(early_stopping=True)
        scoring = "accuracy"

    pipeline = Pipeline(
        steps=[
            (
                "tv1",
                TableVectorizer(datetime=DatetimeEncoder(periodic_encoding="circular")),
            ),
            ("tv2", TableVectorizer(numeric=num_transformer)),
            ("model", model),
        ]
    )
    # pipeline = Pipeline(steps=[('tv1',
    #                             TableVectorizer(
    #                                 datetime=DatetimeEncoder(add_total_seconds=False,
    #                                 periodic_encoding='circular'),
    #                                 numeric=num_transformer)),
    #                            ('model', model)])

    # print(f'{data.X=}')

    # tfmd = TableVectorizer(
    #     numeric=StandardScaler(),
    #     high_cardinality=GapEncoder(3),
    #     datetime=DatetimeEncoder(add_total_seconds=False,
    #       periodic_encoding="circular"),
    # ).fit_transform(data.X)

    # for col in tfmd.columns:
    #     print(tfmd[col])

    scores = cross_val_score(
        pipeline, data.X.iloc[:5000], data.y.iloc[:5000], cv=3, scoring=scoring
    )

    # print(f'Score for {num_transformer.__class__.__name__}: {np.mean(scores)}')
    print(
        f"Score for {num_transformer.__class__.__name__}: {np.mean(scores):g} +-"
        f" {np.std(scores):g}"
    )
    print(f"Scores for {num_transformer.__class__.__name__}: {scores}")


evaluate(StandardScaler())
evaluate(AdaptiveSquashingTransformer())
