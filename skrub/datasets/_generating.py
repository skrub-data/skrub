"""
Functions that generate example data.

"""

from __future__ import annotations

import numbers
import string
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import Bunch, check_random_state


def make_deduplication_data(
    examples,
    entries_per_example,
    prob_mistake_per_letter=0.2,
    random_state=None,
):
    """Duplicates examples with spelling mistakes.

    Characters are misspelled with probability `prob_mistake_per_letter`.

    Parameters
    ----------
    examples : list of str
        Examples to duplicate.
    entries_per_example : list of int
        Number of duplications per example.
    prob_mistake_per_letter : float in [0, 1], default=0.2
        Probability of misspelling a character in duplications.
        By default, 1/5 of the characters will be misspelled.
    random_state : int, RandomState instance, optional
        Determines random number generation for dataset noise. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    list of str
        List of duplicated examples with spelling mistakes.

    Examples
    --------
    >>> from skrub.datasets import make_deduplication_data
    >>> make_deduplication_data(["string1", "string2"],
    ...                         entries_per_example=[4, 5],
    ...                         random_state=9)
    ['btrwng1', 'string1',
    'string1', 'string1',
    'saoing2', 'string2',
    'string2', 'string2',
    'string2']
    """
    rng = check_random_state(random_state)

    data = []
    for example, n_ex in zip(examples, entries_per_example):
        len_ex = len(example)
        # Generate a 2D array of chars of size (n_ex, len_ex)
        str_as_list = np.array([list(example)] * n_ex)
        # Randomly choose which characters are misspelled
        mis_idx = np.where(
            rng.random(len(example[0]) * n_ex) < prob_mistake_per_letter
        )[0]
        # Randomly pick with which character to replace
        replacements = [
            string.ascii_lowercase[i]
            for i in rng.choice(np.arange(26), len(mis_idx)).astype(int)
        ]
        # Introduce spelling mistakes at right examples and char locations per example
        str_as_list[mis_idx // len_ex, mis_idx % len_ex] = replacements
        # go back to 1d array of strings
        data.append(np.ascontiguousarray(str_as_list).view(f"U{len_ex}").ravel())
    return np.concatenate(data).tolist()


def toy_orders(split="train"):
    """Create a toy dataframe and corresponding targets for examples.

    Parameters
    ----------
    split : str, default="train"
        The split to load. Can be either "train", "test", or "all".

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        A dictionary-like object with the keys 'X', 'y' and 'orders'.

    Examples
    --------
    >>> from skrub.datasets import toy_orders
    >>> toy_orders(split="train").X
    ID	product	quantity	date
    0	1	pen	2	2020-04-03
    1	2	cup	3	2020-04-04
    2	3	cup	5	2020-04-04
    3	4	spoon	1	2020-04-05
    >>> toy_orders(split="train").y
    0    False
    1    False
    2     True
    3    False
    Name: delayed, dtype: bool

    If you want both X and y in a dataframe, use `.orders`:

    >>> toy_orders().orders
    ID	product	quantity	date	delayed
    0	1	pen	2	2020-04-03	False
    1	2	cup	3	2020-04-04	False
    2	3	cup	5	2020-04-04	True
    3	4	spoon	1	2020-04-05	False
    """
    X = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6],
            "product": ["pen", "cup", "cup", "spoon", "cup", "fork"],
            "quantity": [2, 3, 5, 1, 5, 2],
            "date": [
                "2020-04-03",
                "2020-04-04",
                "2020-04-04",
                "2020-04-05",
                "2020-04-11",
                "2020-04-12",
            ],
        }
    )
    y = pd.Series([False, False, True, False, True, False], name="delayed")
    if split == "train":
        X, y = X.iloc[:4], y.iloc[:4]
    elif split == "test":
        X, y = X.iloc[4:], y.iloc[4:]
    else:
        assert split == "all", split
    return Bunch(X=X, y=y, orders=X.assign(delayed=y), orders_=X, delayed=y)


def toy_products():
    return pd.DataFrame(
        {
            "description": [
                "screen",
                "hammer",
                "keyboard",
                "usb key",
                "charger",
                "screwdriver",
            ],
            "price": [
                100,
                15,
                20,
                9,
                13,
                12,
            ],
            "seller": [
                "supermarket.com",
                "bestproducts.com",
                "supermarket.com",
                "bestproducts.com",
                "bestproducts.com",
                "supermarket.com",
            ],
            "category": [
                "electronics",
                "tools",
                "electronics",
                "electronics",
                "electronics",
                "tools",
            ],
        }
    )


def toy_cities(seed=0, size=1000, nulls=0.1, n_metrics=4):
    """Generate a synthetic dataframe example with a variety of column types.

    This can be used to showcase dataframes containing strings,
    dates and floats, columns containing null values, and strongly
    correlated columns.

    Contains the following columns:
    uid: A random identifying string of characters.
    cities: A city randomly picked in a list of 20, or a null value.
    encoded_cities: Ordinal encoding applied to the previous column.
    start: A datetime.
    end: A datetime later than the previous, or a null value.
    metric_1, metric_2, etc: Randomly chosen float values.

    Parameters
    ----------
    seed : int, default=0
        Seed for random generation.
    size : int, default=1000
        Number of rows in the output.
    nulls : float in [0, 1], default=0.1
        Probability of a cell in 'cities' or 'end' being null.
    n_metrics : int, default=4
        Number of 'metrics' columns added.

    Returns
    -------
    pandas dataframe
        The randomly-generated dataframe, with `size` rows and
        `5 + n_metrics` columns.

    Examples
    --------
    >>> from skrub.datasets import toy_cities
    >>> df = toy_cities(seed=5, size=3, n_metrics=2)
    >>> df # doctest: +SKIP
              uid     cities  ...  metric_0  metric_1
    0  IPbQyAGoYc  Stockholm  ...  0.227319  0.895448
    1  otDvgcachZ     Vienna  ...  0.872195  0.018517
    2  jHNmownYjU        NaN  ...  0.707496  0.001200
    """

    # Check that the nulls probability is valid
    if isinstance(nulls, bool) or not (isinstance(nulls, numbers.Number)):
        raise ValueError(f"nulls must be a number, got {nulls}.")
    elif not 0 <= nulls <= 1:
        raise ValueError(f"nulls must be a number between 0 and 1, got {nulls!r}.")

    # Check that the other variables are integers
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a positive integer, got {seed}.")
    if not isinstance(size, int) or size < 0:
        raise ValueError(f"size must be a positive integer, got {size}.")
    if not isinstance(n_metrics, int) or n_metrics < 0:
        raise ValueError(f"n_metrics must be a positive integer, got {n_metrics}.")

    rng = np.random.default_rng(seed=seed)
    now = datetime.fromisoformat("2024-01-01").timestamp()
    capitals = [
        "Amsterdam",
        "Athens",
        "Berlin",
        "Bucharest",
        "Budapest",
        "Copenhagen",
        "Dublin",
        "Helsinki",
        "London",
        "Madrid",
        "Paris",
        "Prague",
        "Riga",
        "Rome",
        "Sofia",
        "Stockholm",
        "Vienna",
        "Vilnius",
        "Warsaw",
        "Zagreb",
    ]

    # The first two columns are randomly constructed using lists.
    d = {}

    d["uid"] = [
        "".join(rng.choice(list(string.ascii_letters), 10)) for _ in range(size)
    ]
    d["cities"] = rng.choice(capitals, size=size)
    df_cities = pd.DataFrame(d)

    # `cities` gets assigned null values, and the ordinal encoder is run.
    p = rng.uniform(0, 1, size=size)
    df_cities["cities"] = df_cities["cities"].where(p >= nulls)
    df_cities["encoded_cities"] = OrdinalEncoder().fit_transform(df_cities[["cities"]])

    # Next, the "start" and "end" datetime columns are constructed.
    s = rng.integers(0, int(now), size=size)
    e = rng.integers(s, np.ones(size) * now)
    v = np.vstack([s, e])

    df_dates = pd.DataFrame(v.T, columns=["start", "end"])
    if hasattr(df_dates, "map"):
        df_dates = df_dates.map(datetime.fromtimestamp)
    else:
        df_dates = df_dates.applymap(datetime.fromtimestamp)
    # As above, "end" sees some of its values set to null.
    p = rng.uniform(0, 1, size=size)
    df_dates["end"] = df_dates["end"].where(p >= nulls)

    # Finally, constructing as many "metrics" float columns as specified.
    metric_cols = [f"metric_{k}" for k in range(n_metrics)]
    metrics_array = rng.random(size=(size, n_metrics))
    df_metrics = pd.DataFrame(metrics_array, columns=metric_cols)

    df = pd.concat((df_cities, df_dates, df_metrics), axis=1)

    return df


def make_retail_events(n_users=200, n_events=5000, random_state=None):
    """Generate a synthetic e-commerce clickstream dataset for classification.

    Each row represents one user interaction event on a retail platform.
    The dataset is designed to showcase :class:`~skrub.SessionEncoder` (which
    groups events into sessions using ``user_id`` and ``timestamp``),
    :class:`~skrub.DatetimeEncoder` (which extracts hour-of-day, day-of-week,
    etc. from ``timestamp``), one-hot encoding of categorical features, and
    scaling of numerical features.

    The binary target ``converted`` indicates whether a purchase occurred
    during the session that contains this event.  All events belonging to the
    same session share the same ``converted`` value (a session either converts
    or it does not).  The probability of conversion is determined at the
    session level by the most intent-rich event type in the session
    (``add_to_cart`` > ``wishlist`` > ``search`` > ``page_view``), the
    dominant device, and the mean price viewed — so the signal is learnable
    directly from the observable features.

    Parameters
    ----------
    n_users : int, default=200
        Number of distinct users in the dataset.

    n_events : int, default=5000
        Approximate total number of events (rows) to generate.  The actual
        count may differ slightly because session sizes are drawn from a
        Poisson distribution.

    random_state : int or RandomState instance, optional
        Controls the random number generation for reproducibility.

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        A dictionary-like object with the following attributes:

        - ``X`` : :class:`~pandas.DataFrame` with columns:

          - ``user_id`` : str — user identifier, suitable for
            ``SessionEncoder(split_by="user_id", ...)``.
          - ``timestamp`` : :class:`~pandas.Timestamp` — event time, suitable
            for ``SessionEncoder(timestamp_col="timestamp", ...)`` and
            :class:`~skrub.DatetimeEncoder`.
          - ``device_type`` : str — one of ``"mobile"``, ``"desktop"``,
            ``"tablet"``; encode with :class:`~sklearn.preprocessing.OneHotEncoder`.
          - ``page_category`` : str — one of ``"electronics"``, ``"fashion"``,
            ``"home"``, ``"sports"``, ``"books"``; encode with
            :class:`~sklearn.preprocessing.OneHotEncoder`.
          - ``event_type`` : str — one of ``"page_view"``, ``"search"``,
            ``"add_to_cart"``, ``"wishlist"``; encode with
            :class:`~sklearn.preprocessing.OneHotEncoder`.
          - ``time_on_page`` : float — seconds spent on the page (exponential
            distribution, mean ≈ 120 s).
          - ``price_viewed`` : float — price of the item viewed (log-normal).

        - ``y`` : :class:`~pandas.Series` of bool, name ``"converted"`` — the
          classification target.

    Examples
    --------
    >>> from skrub.datasets import make_retail_events
    >>> bunch = make_retail_events(n_users=20, n_events=100, random_state=0)
    >>> bunch.X.shape[1]  # 7 feature columns; rows ≈ n_events
    7
    >>> bunch.X.columns.tolist()
    ['user_id', 'timestamp', 'device_type', 'page_category', 'event_type', 'time_on_page', 'price_viewed']
    >>> bunch.y.name
    'converted'
    >>> bunch.y.dtype
    dtype('bool')
    """  # noqa: E501
    rng = check_random_state(random_state)

    # --- users -----------------------------------------------------------
    user_ids = [f"user_{i:04d}" for i in range(n_users)]

    # Distribute events across users with a power-law (Pareto) weight so that
    # a small number of users generate the bulk of the activity.
    activity_weights = rng.pareto(2.0, size=n_users) + 1.0
    activity_weights /= activity_weights.sum()
    events_per_user = rng.multinomial(n_events, activity_weights)

    # --- timestamps with realistic session structure ---------------------
    # Events form *sessions*: bursts of activity where consecutive events are
    # only ~90 s apart, separated by long idle gaps (at least 2 h).  This
    # structure is what SessionEncoder is designed to detect.
    #
    # For each user:
    #   1. Split their event budget into sessions of Poisson(3)+1 events.
    #   2. Space session starts by Exponential gaps >> session_gap, spread
    #      across a 90-day window.
    #   3. Within each session, place events with Exponential(90 s) gaps.
    base_time = pd.Timestamp("2024-01-01")
    total_window_s = 90 * 24 * 3600  # 90 days
    within_session_mean_s = 90.0  # ~1.5 min between events inside a session
    min_between_session_s = 2 * 3600  # 2 h minimum gap — well above session_gap

    all_user_ids: list = []
    all_timestamps: list = []
    all_session_keys: list = []  # unique key per (user, session) pair

    for uid, n_user_events in zip(user_ids, events_per_user):
        if n_user_events == 0:
            continue

        # Split into sessions; each session has at least 1 event.
        session_sizes = []
        remaining = int(n_user_events)
        while remaining > 0:
            size = min(int(rng.poisson(3)) + 1, remaining)
            session_sizes.append(size)
            remaining -= size

        n_sessions = len(session_sizes)

        # Session start times: inter-session gaps drawn from Exponential so
        # that they are spread over the 90-day window but always exceed the
        # minimum between-session gap.
        mean_gap_s = max(total_window_s / n_sessions, min_between_session_s)
        inter_gaps = rng.exponential(scale=mean_gap_s, size=n_sessions)
        # Random offset for the very first session start.
        inter_gaps[0] += rng.uniform(0, min_between_session_s)
        session_starts_s = np.cumsum(inter_gaps)

        for sess_idx, (start_s, sess_size) in enumerate(
            zip(session_starts_s, session_sizes)
        ):
            # Events are placed at start_s, start_s+gap1, start_s+gap1+gap2 …
            within_gaps = np.concatenate(
                [[0.0], rng.exponential(within_session_mean_s, size=sess_size - 1)]
            )
            session_key = f"{uid}_{sess_idx}"
            for offset_s in start_s + np.cumsum(within_gaps):
                all_user_ids.append(uid)
                all_timestamps.append(base_time + pd.Timedelta(seconds=float(offset_s)))
                all_session_keys.append(session_key)

    n_actual = len(all_user_ids)

    # --- categorical features --------------------------------------------
    device_type = rng.choice(
        ["mobile", "desktop", "tablet"],
        size=n_actual,
        p=[0.55, 0.35, 0.10],
    )
    page_category = rng.choice(
        ["electronics", "fashion", "home", "sports", "books"],
        size=n_actual,
    )
    event_type = rng.choice(
        ["page_view", "search", "add_to_cart", "wishlist"],
        size=n_actual,
        p=[0.60, 0.20, 0.15, 0.05],
    )

    # --- numerical features ----------------------------------------------
    # time_on_page: seconds spent on page (heavy-tailed)
    time_on_page = rng.exponential(scale=120.0, size=n_actual).round(1)
    # price_viewed: item price in USD (log-normal, median ≈ e^3.5 ≈ 33)
    price_viewed = np.exp(rng.normal(loc=3.5, scale=1.2, size=n_actual)).round(2)

    # --- assemble & sort -------------------------------------------------
    X = pd.DataFrame(
        {
            "user_id": all_user_ids,
            "timestamp": all_timestamps,
            "_session_key": all_session_keys,
            "device_type": device_type,
            "page_category": page_category,
            "event_type": event_type,
            "time_on_page": time_on_page,
            "price_viewed": price_viewed,
        }
    )
    # Sorting by timestamp
    X = X.sort_values(["timestamp"]).reset_index(drop=True)

    # --- target: converted (bool), assigned per session ------------------
    # A session either converts or it does not — all events in a session
    # share the same label.  This is the realistic framing: a checkout
    # either happens in a session or it doesn't.
    #
    # The conversion probability is a logistic function of session-level
    # summaries of observable features, so the model can learn it:
    #
    #   best_event : the most purchase-intent event type in the session
    #               (add_to_cart >> wishlist >> search >> page_view)
    #   device     : dominant device (desktop > tablet > mobile)
    #   price      : mean price viewed (expensive items convert less)
    event_intent = X["event_type"].map(
        {"add_to_cart": 2.0, "wishlist": 0.8, "search": 0.0, "page_view": -0.5}
    )
    device_score_col = X["device_type"].map(
        {"desktop": 0.5, "tablet": 0.1, "mobile": -0.3}
    )
    price_score_col = -0.2 * np.log1p(X["price_viewed"])

    tmp = X[["_session_key"]].assign(
        event_intent=event_intent,
        device_score=device_score_col,
        price_score=price_score_col,
    )
    session_logits = (
        tmp.groupby("_session_key")
        .agg(
            event_intent=("event_intent", "max"),
            device_score=("device_score", "mean"),
            price_score=("price_score", "mean"),
        )
        .sum(axis=1)
    )

    # Add one noise draw per session (not per event) so the label is
    # consistent within a session.
    unique_keys = session_logits.index.tolist()
    noise = dict(zip(unique_keys, rng.normal(0.0, 0.5, size=len(unique_keys))))
    session_prob = {
        k: 1.0 / (1.0 + np.exp(-(session_logits[k] + noise[k]))) for k in unique_keys
    }
    session_converted = {k: bool(rng.binomial(1, session_prob[k])) for k in unique_keys}

    y = X["_session_key"].map(session_converted).rename("converted").astype(bool)
    X = X.drop(columns=["_session_key"])

    return Bunch(X=X, y=y)
