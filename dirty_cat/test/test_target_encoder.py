import numpy as np

from dirty_cat import target_encoder


def test_similarity_encoder():
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
