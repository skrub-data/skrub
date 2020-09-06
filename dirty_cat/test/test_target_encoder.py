import numpy as np
import pytest

from dirty_cat import target_encoder


def test_target_encoder():
    lambda_ = target_encoder.lambda_
    X1 = np.array(['Red',
                   'red',
                   'green',
                   'blue',
                   'green',
                   'green',
                   'blue',
                   'red']).reshape(-1, 1)
    X2 = np.array(['male',
                   'male',
                   'female',
                   'male',
                   'female',
                   'female',
                   'female',
                   'male']).reshape(-1, 1)
    X = np.hstack([X1, X2])

    # Case 1: binary-classification and regression
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    n = len(y)

    Ey_ = 3/8
    Eyx_ = {'color': {'Red': 1,
                      'red': 0,
                      'green': 1/3,
                      'blue': .5},
            'gender': {'male': .5,
                       'female': .25}}
    count_ = {'color': {'Red': 1,
                        'red': 2,
                        'green': 3,
                        'blue': 2},
              'gender': {'male': 4,
                         'female': 4}}

    encoder = target_encoder.TargetEncoder()
    encoder.fit(X, y)
    for j in range(X.shape[1]):
        assert np.array_equal(encoder.categories_[j], np.unique(X[:, j]))
    assert Ey_ == encoder.Ey_
    assert encoder.Eyx_[0] == Eyx_['color']
    assert encoder.Eyx_[1] == Eyx_['gender']

    Xtest1 = np.array(['Red',
                       'red',
                       'blue',
                       'green',
                       'Red',
                       'red',
                       'blue',
                       'green']).reshape(-1, 1)
    Xtest2 = np.array(['male',
                       'male',
                       'male',
                       'male',
                       'female',
                       'female',
                       'female',
                       'female']).reshape(-1, 1)
    Xtest = np.hstack([Xtest1, Xtest2])

    Xout = encoder.transform(Xtest)

    ans_dict = {var:
                {cat:
                 Eyx_[var][cat] *
                 lambda_(count_[var][cat], n/len(count_[var])) +
                 Ey_ * (1 - lambda_(count_[var][cat], n/len(count_[var])))
                 for cat in Eyx_[var]}
                for var in Eyx_
                }
    ans = np.zeros((n, 2))
    for j, col in enumerate(['color', 'gender']):
        for i in range(n):
            ans[i, j] = ans_dict[col][Xtest[i, j]]
    assert np.array_equal(Xout, ans)

    # Case 2: multiclass-classification
    y = np.array([1, 0, 2, 1, 0, 1, 0, 0])
    n = len(y)

    encoder = target_encoder.TargetEncoder(clf_type='multiclass-clf')
    encoder.fit(X, y)

    Ey_ = {0: 4/8, 1: 3/8, 2: 1/8}
    Eyx_ = {0: {}, 1: {}, 2: {}}
    Eyx_[0] = {'color': {'Red': 0,
                         'red': 1,
                         'green': 1/3,
                         'blue': .5},
               'gender': {'male': .5,
                          'female': .5}}
    Eyx_[1] = {'color': {'Red': 1,
                         'red': 0,
                         'green': 1/3,
                         'blue': .5},
               'gender': {'male': .5,
                          'female': .25}}
    Eyx_[2] = {'color': {'Red': 0,
                         'red': 0,
                         'green': 1/3,
                         'blue': 0},
               'gender': {'male': 0,
                          'female': .25}}
    assert np.array_equal(np.unique(y), encoder.classes_)
    for k in [0, 1, 2]:
        assert Ey_[k] == encoder.Ey_[k]
        assert encoder.Eyx_[k][0] == Eyx_[k]['color']
        assert encoder.Eyx_[k][1] == Eyx_[k]['gender']

    count_ = {'color': {'Red': 1,
                        'red': 2,
                        'green': 3,
                        'blue': 2},
              'gender': {'male': 4,
                         'female': 4}}
    assert count_['color'] == encoder.counter_[0]
    assert count_['gender'] == encoder.counter_[1]

    ans_dict = {0: {}, 1: {}, 2: {}}
    for k in [0, 1, 2]:
        ans_dict[k] = {var:
                       {cat:
                        Eyx_[k][var][cat] *
                        lambda_(count_[var][cat], n/len(count_[var])) +
                        Ey_[k] *
                        (1 - lambda_(count_[var][cat], n/len(count_[var])))
                        for cat in Eyx_[k][var]}
                       for var in Eyx_[k]
                       }

    ans = np.zeros((n, 2*3))
    Xtest1 = np.array(['Red',
                       'red',
                       'blue',
                       'green',
                       'Red',
                       'red',
                       'blue',
                       'green']).reshape(-1, 1)
    Xtest2 = np.array(['male',
                       'male',
                       'male',
                       'male',
                       'female',
                       'female',
                       'female',
                       'female']).reshape(-1, 1)
    Xtest = np.hstack([Xtest1, Xtest2])
    for k in [0, 1, 2]:
        for j, col in enumerate(['color', 'gender']):
            for i in range(n):
                ans[i, 2*j+k] = ans_dict[k][col][Xtest[i, j]]

    Xout = encoder.transform(Xtest)
    ans = np.zeros((n, 2*3))
    Xtest1 = np.array(['Red',
                       'red',
                       'blue',
                       'green',
                       'Red',
                       'red',
                       'blue',
                       'green']).reshape(-1, 1)
    Xtest2 = np.array(['male',
                       'male',
                       'male',
                       'male',
                       'female',
                       'female',
                       'female',
                       'female']).reshape(-1, 1)
    Xtest = np.hstack([Xtest1, Xtest2])
    for k in [0, 1, 2]:
        for j, col in enumerate(['color', 'gender']):
            for i in range(n):
                ans[i, j*3+k] = ans_dict[k][col][Xtest[i, j]]
    encoder = target_encoder.TargetEncoder(clf_type='multiclass-clf')
    encoder.fit(X, y)
    Xout = encoder.transform(Xtest)
    assert np.array_equal(Xout, ans)


def _test_missing_values(input_type, missing):
    lambda_ = target_encoder.lambda_
    X = [['Red', 'male'],
         [np.nan, 'male'],
         ['green', 'female'],
         ['blue', 'male'],
         ['green', 'female'],
         ['green', 'female'],
         ['blue', 'female'],
         [np.nan, np.nan]]

    color_cat = ['Red', '', 'green', 'blue']
    gender_cat = ['male', '', 'female']

    if input_type == 'numpy':
        X = np.array(X, dtype=object)
    elif input_type == 'pandas':
        pd = pytest.importorskip("pandas")
        X = pd.DataFrame(X)

    # Case 1: binary-classification and regression
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    n = len(y)

    Ey_ = 3/8
    Eyx_ = {'color': {'Red': 1,
                      '': 0,
                      'green': 1/3,
                      'blue': .5},
            'gender': {'male': 2/3,
                       'female': .25,
                       '': 0}}
    count_ = {'color': {'Red': 1,
                        '': 2,
                        'green': 3,
                        'blue': 2},
              'gender': {'male': 3,
                         'female': 4,
                         '': 1}}

    encoder = target_encoder.TargetEncoder(handle_missing=missing)
    if missing == 'error':
        with pytest.raises(ValueError, match=r"Found missing values in input "
                           "data; set handle_missing='' to encode "
                           "with missing values"):
            encoder.fit_transform(X, y)
        return
    elif missing == '':
        encoder.fit_transform(X, y)

        assert set(encoder.categories_[0]) == set(color_cat)
        assert set(encoder.categories_[1]) == set(gender_cat)
        assert Ey_ == encoder.Ey_
        assert encoder.Eyx_[0] == Eyx_['color']
        assert encoder.Eyx_[1] == Eyx_['gender']
        assert dict(encoder.counter_[0]) == count_['color']
        assert dict(encoder.counter_[1]) == count_['gender']
    else:
        with pytest.raises(ValueError, match=r"handle_missing"
                           " should be either 'error' or ''"):
            encoder.fit_transform(X, y)
        return


def _test_missing_values_transform(input_type, missing):
    lambda_ = target_encoder.lambda_
    X = [['Red', 'male'],
         ['red', 'male'],
         ['green', 'female'],
         ['blue', 'male'],
         ['green', 'female'],
         ['green', 'female'],
         ['blue', 'female'],
         ['red', 'male']]

    color_cat = ['Red', 'red', 'green', 'blue']
    gender_cat = ['male', 'female']

    X_test = [['Red', 'male'],
              [np.nan, 'male'],
              ['green', 'female'],
              ['blue', 'male'],
              ['green', 'female'],
              ['green', 'female'],
              ['blue', 'female'],
              [np.nan, np.nan]]

    if input_type == 'numpy':
        X_test = np.array(X_test, dtype=object)
    elif input_type == 'pandas':
        pd = pytest.importorskip("pandas")
        X_test = pd.DataFrame(X_test)

    # Case 1: binary-classification and regression
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    n = len(y)

    Ey_ = 3/8
    Eyx_ = {'color': {'Red': 1,
                      'red': 0,
                      'green': 1/3,
                      'blue': .5},
            'gender': {'male': .5,
                       'female': .25}}
    count_ = {'color': {'Red': 1,
                        'red': 2,
                        'green': 3,
                        'blue': 2},
              'gender': {'male': 4,
                         'female': 4}}

    encoder = target_encoder.TargetEncoder(handle_unknown='ignore',
                                           handle_missing=missing)
    if missing == 'error':
        encoder.fit_transform(X, y)
        with pytest.raises(ValueError, match=r"Found missing values in input "
                           "data; set handle_missing='' to encode "
                           "with missing values"):
            encoder.transform(X_test)
        return
    elif missing == '':
        encoder.fit_transform(X, y)
        ans = encoder.transform(X_test)

        assert np.allclose(ans[1, 0], Ey_)
        assert np.allclose(ans[-1, 0], Ey_)


def test_missing_values():
    input_types = ['list', 'numpy', 'pandas']
    handle_missing = ['aaa', 'error', '']
    for input_type in input_types:
        for missing in handle_missing:
            _test_missing_values(input_type, missing)
            _test_missing_values_transform(input_type, missing)
