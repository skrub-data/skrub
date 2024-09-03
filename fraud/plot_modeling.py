"""
=========================
BNP Fraudsters - Modeling
=========================

The previous EDA notebook detailed how different columns in this dataset are linked, in
particular the hierarchical relationships between our categorical columns
``item, make`` < ``model`` < ``goods_code`` < ``ID``.

We also noticed that most categorical columns showed morphological similarities and
typos, which increases the cardinality of each column significantly. When not properly
encoded, high cardinality columns usually have few samples for each category, degrading
the predictive performance of the estimator.

We also established that each row in this dataset conceptually represents a basket,
which is a list of products of variable length. The dataframe representation flattens
these lists by fixing the number of products to 24, which creates a sparse dataframe.

The products order in the list might also bring predictive power, at the cost of
reducing the vocabulary for each categorical column, since some items will never been
observed for some position in the basket.

Finally, as we are tackling a fraudulent loans detection use-case, the dataset is
extremely imbalanced, with a prevalence around 1.4%. We will try to derive a cost
function based on few questionable assumptions about the use-case. To improve this
function, it is mandatory to get more insights from a domain expert at BNP Paribas and
contextualise the working environment of the model and how its predictions will be used.

Setup
-----

We start with loading the features and showing the attributes of the first product in
each basket.
"""

# %%
import numpy as np
import pandas as pd
import seaborn as sns
from utils_eda import get_group_cols

X = pd.read_parquet("bnp_fraud.parquet")
X.pop("ID")

renaming = dict(
    zip(
        get_group_cols("Nbr_of_prod_purchas"),
        get_group_cols("nbr"),
    )
)
X = X.rename(columns=renaming)

attribute_cols = ["item", "cash_price", "make", "model", "goods_code", "nbr"]
first_cols = [*[f"{col}1" for col in attribute_cols], "Nb_of_items"]
X[first_cols]

# %%
# Note that the full dataframe contains 24 times these columns:
X.shape

# %%
# We load the target and apply a stratified shuffle split to get a training, validation
# and testing dataset, with ratios 81% / 9% / 10%. In this notebook, the validation set
# will be used for post-training tuning of the classification threshold. This could also
# be used for e.g. calibrating trained estimators.
#
# Note that for computation cost reasons, **we don't use an outer cross validation loop
# here**. This would consist in iterating over various random seeds during the split
# operation, and estimating the performance across these different splits. This would
# give us a stronger estimate of our models performance and confidence intervals.

y = X.pop("fraud_flag")
y

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    stratify=y,
    random_state=0,
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.1,
    stratify=y_train,
    random_state=0,
)
X_train.shape, X_val.shape, X_test.shape


# %%
# We need to define scoring metrics to estimate the performance of our models.
#
# - Our first reflex should always be to **create a utility metric with a tangible
#   unit**. In our case, it could be the gains yielded by our model in euros, defined
#   by a cost matrix.
# - It's also useful to check **proper scoring rules** on the predicted probabilities,
#   like the log loss and the brier score. In general terms, we should avoid optimizing
#   thresholded metrics like accuracy or f1-score during training, as the default
#   threshold of 0.5 is often not adapted to our problem from a utility point of view.
#
# We define our cost matrix using the total price of the basket. Our assumptions are:
#
# - A **true positive** prediction yields 0€. We identified a fraudulent loan and
#   rejected it. No harm has been done.
# - A **true negative** prediction yields 7% of the total price. This corresponds to a
#   premium applied by the creditor on the loan amount, on a non fraudulent transaction.
# - A **false positive** prediction yields a fix gain of -5€. This is the price of a
#   false alarm, maybe in reputation or using resources to run a KYC and finally having
#   an agent deciding whether to accept the loan or not.
# - A **false negative** is the more costly error, since the fraudster gets away with
#   the total price of the basket. This corresponds to a net loss of the total price.
#
# We define the total basket price as the sum of the products unitary prices, multiplied
# by their respective quantities.

MAX_ITEMS = 24


def total_price(X):
    total_price = pd.Series(np.zeros(X.shape[0]), index=X.index, name="total_price")

    for idx in range(1, MAX_ITEMS + 1):
        total_price += X[f"cash_price{idx}"].fillna(0) * X[f"nbr{idx}"].fillna(0)

    return total_price


# %%
# We compute the total price over the three datasets. Note that we could have also
# computed this quantity before splitting.

for X_ in [X_train, X_val, X_test]:
    X_["total_price"] = total_price(X_)

# %%
# We use the ``make_scorer`` function to turn a metric into a score that can be used by
# a cross validation object in scikit-learn. For proper scoring rules, lower is better,
# and we need to use predicted classes probabilities instead of predicted classes.
#
# We also need to activate metadata routing to compute the cost gain, since we need to
# pass the ``total_price`` column to the score function.

import sklearn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    make_scorer,
    precision_score,
    recall_score,
)


def fpr_score(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    tn, fp, _, _ = cm.ravel()
    return fp / (tn + fp)


def credit_gain_score(y_true, y_pred, amount):
    mask_tn = (y_true == 0) & (y_pred == 0)
    mask_fp = (y_true == 0) & (y_pred == 1)
    mask_fn = (y_true == 1) & (y_pred == 0)

    fraudulent_refuse = 0
    fraudulent_accept = -amount[mask_fn].sum()
    legitimate_refuse = mask_fp.sum() * -5
    legitimate_accept = (amount[mask_tn] * 0.07).sum()

    return fraudulent_refuse + fraudulent_accept + legitimate_refuse + legitimate_accept


tpr_score = recall_score  # TPR and recall are the same metric
sklearn.set_config(enable_metadata_routing=True)
scoring = {
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
    "fpr": make_scorer(fpr_score),
    "tpr": make_scorer(tpr_score),
    "log_loss": make_scorer(
        log_loss, greater_is_better=False, response_method="predict_proba"
    ),
    "brier_score_loss": make_scorer(
        brier_score_loss, greater_is_better=False, response_method="predict_proba"
    ),
    "cost_gain": make_scorer(credit_gain_score).set_score_request(amount=True),
    "average_precision_score": make_scorer(average_precision_score),
}

results = dict()


# %%
# We write a simple utility function to make a result dictionary for each model,
# containing the predicted probabilities, and all the score we defined above.


def get_results(model, X_test, y_test, amount, threshold):
    results_ = {
        "y_proba": model.predict_proba(X_test),
        "y_test": y_test,
        "model": model,
        "threshold": threshold,
        "params": model.get_params(),
    }
    for scoring_name, scoring_func in scoring.items():
        if scoring_name == "cost_gain":
            results_[scoring_name] = scoring_func(model, X_test, y_test, amount=amount)
        else:
            score = scoring_func(model, X_test, y_test)
            if scoring_name in ["brier_score_loss", "log_loss"]:
                score *= -1
            results_[scoring_name] = score

    return results_


# %%
# The setup is now done. Let's start evaluating models!
#
# Dummy baselines and oracle
# --------------------------
#
# On one hand, we have to score **dummy baselines**. Dummies are by definition the
# lowest effort possible. It would be concerning to not beat them. We use two distinct
# strategies:
#
# #. Randomly predict the target using the prevalence i.e. predicting 1 only 1.4% of
#    the time (``stratified``)
# #. Always return 1 (``constant``).
#
# On the other hand, we also create **an oracle**, i.e. a classifier with perfect
# predictions. It is generally impossible to attain due to the non explainable variance
# in the target. The oracle classifier simply saves the target of the test set and
# returns it.
#
# These two type of models form our expected performance lower and upper bounds.

from sklearn.dummy import DummyClassifier
from utils_oracle import OracleClassifier

dummy_random = DummyClassifier(strategy="stratified").fit(X_train, y_train)

dummy_negative = DummyClassifier(strategy="constant", constant=0).fit(X_train, y_train)

oracle = OracleClassifier().fit(None, y_test)

results["stratified_random"] = get_results(
    dummy_random, X_test, y_test, amount=X_test["total_price"], threshold=0.5
)
results["always_negative"] = get_results(
    dummy_negative, X_test, y_test, amount=X_test["total_price"], threshold=0.5
)
results["oracle"] = get_results(
    oracle, X_test, y_test, amount=X_test["total_price"], threshold=0.5
)

# %%
# Low effort estimator
# --------------------
#
# Once these baselines are in place, we can start proper modeling. The first idea that
# comes to mind is to limit our feature engineering to a simple encoding of our
# categories with a ``HistGradientBoostingClassifier``. This is our low effort
# estimator.
#
# As the cardinality of most categorical columns is higher than the number of bins
# created by the boosting tree (255), we have some few encoding options:
#
# - **Reducing the categories to the top 254 most frequent**, with an extra categories
#   for the NaN and the unseen during predict. We can do so with an OrdinalEncoder, but
#   we would loose some of the rare categories in an "other" bin.
# - **Encoding the categories as float**. This can be done efficiently with the
#   ``TargetEncoder`` and is our favourite option.
#
# The ``TargetEncoder`` also creates some sort of similarity between different
# categories that have the same target frequency, whereas the ``OrdinalEncoder``
# considers all categories as orthogonal.
#
# Note that we would prefer ``OrdinalEncoder`` to ``OneHotEncoder`` because:
#
# #. Creating one column per categories with a ``OneHotEncoder`` is impracticable with
#    this dataset and the dimensionality would explode.
# #. Boosting tree are natively able to handle categories, and unlike linear models,
#    the ordering of the ``OrdinalEncoder`` doesn't affect tree-based models. Plus,
#    creating additional columns with a ``OneHotEncoder`` usually diminishes the
#    inductive biases of the trees.
#
# Last but not least, boosting tree models natively handles missing values by
# considering them as another category, so we don't even need to impute them.
#
# We use the ``TableVectorizer`` from skrub which automatically selects the categorical
# columns (string, object or categorical dtypes in pandas) and apply the
# ``TargetEncoder`` on them. We don't transform numeric columns.

from sklearn.preprocessing import TargetEncoder
from skrub import TableVectorizer

vectorizer = TableVectorizer(
    high_cardinality=TargetEncoder(),
    cardinality_threshold=1,
)
vectorizer

# %%
# After defining our vectorizer, our next aim is to fine tune the hyper-parameters of
# our boosting tree. Since fitting the vectorizer takes approximately 10s and making
# prediction 1s, we don't chain the vectorizer and the boosting tree together in a
# pipeline to limit redundant computations during cross validation.
#
# Note that we don't use the ``memory`` argument of the pipeline and precompute instead
# the vectorized dataset only once to save even more time.

X_trans_train = vectorizer.fit_transform(X_train, y_train)
X_trans_val = vectorizer.transform(X_val)
X_trans_test = vectorizer.transform(X_test)
X_trans_train

# %%
# We use a ``RandomizedSearchCV`` to find the best hyper-parameters combination, over a
# small quantity of draws to stay computationally cheap (``n_iter=5``).
#
# The model itself is trained on the log loss (the default), so **we evaluate it on
# the brier score** to avoid training and evaluating using the same metric. Doing
# otherwise can lead to over-optimistic estimation of the performance.

from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

param_distributions = dict(
    learning_rate=loguniform(0.001, 0.5),
    max_depth=randint(2, 10),
    min_samples_leaf=randint(5, 50),
    max_iter=randint(100, 1000),
)

random_search_params = dict(
    param_distributions=param_distributions,
    n_iter=5,
    scoring=scoring["brier_score_loss"],
    refit=True,
    verbose=1,
    error_score="raise",
)

low_effort_hgbt = HistGradientBoostingClassifier()

hgbt_search = RandomizedSearchCV(
    low_effort_hgbt,
    **random_search_params,
).fit(X_trans_train, y_train)

# %%
# We then evaluate the performance on the test set using our helper function, and
# display the brier score and the cost gain.

from utils_modeling import plot_metric

results["low_effort_hgbt"] = get_results(
    hgbt_search.best_estimator_,
    X_trans_test,
    y_test,
    amount=X_test["total_price"],
    threshold=0.5,
)

plot_metric(results, metric_name="brier_score_loss")

# %%
# We beat the dummy baselines by a slight margin, and are quite far from the oracle at
# 0.
#
# Let's compare this to the log loss.

plot_metric(results, metric_name="log_loss")

# %%
# Since the model has been trained on the log loss, it is significantly better than the
# dummies.
#
# Next, we should look at our utility function. Is our model significantly better than
# the dummies on this metric too?

plot_metric(results, metric_name="cost_gain")

# %%
# The gap between our model and the dummies for the utility function is tighter, like
# for the brier score. A model always accepting all transactions yields 638k€, while
# the best achievable gain of the oracle is around 831k€. Our model gain is 639€k.
# Error bars would be helpful here.
#
# However, the gain has been evaluated with the default classification threshold of 0.5,
# so there is room for improvement by optimizing this threshold using the
# ``TunedThresholdClassifierCV``. This time, **we set the scoring objective as the cost
# gain**.
#
# To avoid running an expensive nested cross validation of the randomized search within
# the tuned threshold classifier, we reuse the previous best fitted model with
# ``cv=prefit``. By doing so, **we must use the validation set instead of the training
# set** to avoid introducing an important bias.

from sklearn.model_selection import TunedThresholdClassifierCV

tuned_params = dict(
    scoring=scoring["cost_gain"],
    store_cv_results=True,
    cv="prefit",
    refit=False,
)

tuned_hgbt = TunedThresholdClassifierCV(
    hgbt_search.best_estimator_,
    **tuned_params,
).fit(
    X_trans_val,
    y_val,
    amount=X_val["total_price"],
)

results["low_effort_hgbt_tuned"] = get_results(
    tuned_hgbt,
    X_trans_test,
    y_test,
    amount=X_test["total_price"],
    threshold=tuned_hgbt.best_threshold_,
)

plot_metric(results, metric_name="cost_gain")


# %%
# Our gain is now higher than 670k€, which is a nice and quite cheap improvement!
#
# A good follow-up is to investigate which hyper-parameters are correlated with a low
# brier score and a high fitting time.

cv_results = pd.DataFrame(hgbt_search.cv_results_["params"])

for col in ["mean_test_score", "mean_fit_time"]:
    cv_results[col] = hgbt_search.cv_results_[col]

cv_results = cv_results.sort_values("mean_test_score", ascending=False)
cv_results

# %%
from matplotlib import pyplot as plt

corr = cv_results.corr()
corr_triu = np.triu(corr)
sns.heatmap(corr, annot=True, mask=corr_triu)
plt.tight_layout()

# %%
# Since we are considering the negative brier score, higher is better.
# **These correlations must be taken with a big grain of salt though** because the
# number of combination tested is  small (``n_iter=5``).
#
# MinHash estimator
# -----------------
#
# How could we improve upon the low effort estimator? We could first try to derive more
# features from each basket. Starting with the numeric columns, a simple idea is to
# compute basic statistics for the unitary price and the quantity of each product.


def row_aggregate_number(X, operations=("mean", "sum", "std", "min", "max")):
    X_out = pd.DataFrame(index=X.index)

    for attribute_col in ["cash_price", "nbr"]:
        cols = get_group_cols(attribute_col)
        for operation in operations:
            X_col = (
                X[cols]
                .apply(operation, axis=1)
                .rename(f"{attribute_col}_{operation}")
                .fillna(0)
            )
            X_out = pd.concat([X_out, X_col], axis=1)

    X_out["total_price"] = total_price(X)

    return X_out


# %%
# Since this function is stateless (i.e. it doesn't require a fitting step), we can
# easily include it in a pipeline using a ``FunctionTransformer``. It takes a dataframe
# as input and output the transformation.

from sklearn.preprocessing import FunctionTransformer

row_aggregate_number_transformer = FunctionTransformer(row_aggregate_number)
row_aggregate_number_transformer.fit_transform(X_train, y_train)

# %%
# Next, let's take a closer look at the categories. We saw in the EDA notebook that
# there were typos and similarities, so we need an encoder able to handle morphological
# variations. Here, we can use both ``MinHashEncoder`` and ``GapEncoder`` from skrub.
# The former is a stateless transformer, fast to compute and whose embeddings are
# efficient for learning with tree based models.
#
# However, with 3 * 24 categorical columns to encode and embeddings of size 30 by
# default, our total output dimension would be 3 * 24 * 30 = 2160 dimensions, which
# would be very expensive to store in RAM.
#
# So we need row-wise aggregation of our embeddings. But how to aggregate these?
# MinHash embeddings are not comparable in an Euclidean space, but they encode minimum
# of some hash values, so we can take **the minimum value** for each dimension across
# all 24 products, for each group (``item``, ``make``, and ``model``).
#
# For each basket, this yields an embedding vector of size 3 * 30.

from skrub import MinHashEncoder


def row_aggregate_minhash(
    X,
    attribute_cols=("item", "make", "model"),
    encoder=None,
):
    if encoder is None:
        encoder = MinHashEncoder()

    X_out = pd.DataFrame(index=X.index)

    for attribute_col in attribute_cols:
        group_cols = get_group_cols(attribute_col)

        X_trans = TableVectorizer(
            high_cardinality=encoder,
            cardinality_threshold=1,  # no low cardinality
        ).fit_transform(X[group_cols], y)

        n_samples, n_columns = X_trans.shape[0], len(group_cols)
        X_trans = (
            X_trans.to_numpy()
            .reshape(n_samples, n_columns, -1)  # (n_samples, 24, 30)
            .min(
                axis=1
            )  # take minimum across the basket, for each embedding dimension.
        )
        columns = [f"{attribute_col}_e{idx}" for idx in range(X_trans.shape[1])]
        X_trans = pd.DataFrame(
            X_trans,
            columns=columns,
            index=X.index,
        )
        X_out = pd.concat([X_out, X_trans], axis=1)

    return X_out


# %%
row_aggregate_minhash_transformer = FunctionTransformer(row_aggregate_minhash)
row_aggregate_minhash_transformer.fit_transform(X_train, y_train)

# %%
# ``goods_code`` however is an ID with no morphological variations, so we don't expect
# MinHash to be efficient here. Instead, we encode each ``goods_code`` column with a
# target encoder, and then compute a row-wise weighted average, using the quantity
# ``nbr`` as weight.
#
# This yields a single column.


def row_aggregate_post_target_encoder(X):
    goods_code_cols = get_group_cols("goods_code")
    nbr_cols = get_group_cols("nbr")
    avg_encoding = np.average(
        X[goods_code_cols],
        weights=X[nbr_cols].fillna(0),
        axis=1,
    )
    return pd.Series(avg_encoding, index=X.index, name="goods_code").to_frame()


# %%
# We use Pipeline, ColumnTransformer and FeatureUnion instead of make_pipeline,
# make_column_transformer and make_union to set meaningful names.

from sklearn.pipeline import Pipeline

target_encoder = TableVectorizer(
    specific_transformers=[(TargetEncoder(), get_group_cols("goods_code"))],
    low_cardinality="passthrough",
    high_cardinality="passthrough",
)

row_aggregate = FunctionTransformer(row_aggregate_post_target_encoder)

row_aggregate_target_encoder = Pipeline(
    [
        ("target_encoder", target_encoder),
        ("row_aggregate", row_aggregate),
    ]
)
row_aggregate_target_encoder


# %%
# The TargetEncoder does not appreciate <NA> from pandas string dtype.
X_train[get_group_cols("goods_code")] = X_train[get_group_cols("goods_code")].astype(
    str
)

row_aggregate_target_encoder.fit_transform(X_train, y_train)

# %%

# %%
# Finally, we can assemble our aggregation vectorizer by unioning all of our
# transformers. This will act as a horizontal concatenation of all transformers outputs.

from sklearn.pipeline import FeatureUnion

vectorizer = FeatureUnion(
    [
        ("agg_number", row_aggregate_number_transformer),
        ("agg_category", row_aggregate_minhash_transformer),
        ("agg_goods_code", row_aggregate_target_encoder),
    ]
).set_output(transform="pandas")

vectorizer

# %%
# Like we said previously, vectorizing the dataframe is expensive when performing cross
# validation, so we prefer computing the transformations in advance.

X_trans_train = vectorizer.fit_transform(X_train, y_train)
X_trans_val = vectorizer.transform(X_val)
X_trans_test = vectorizer.transform(X_test)
X_trans_train

# %%
# We use the same boosting tree estimator. We run a randomized search with the same
# budget and hyper parameter distributions, before tuning the threshold on the
# validation set and evaluating performance on the test set.

minhash_hgbt = HistGradientBoostingClassifier()

minhash_hgbt_search = RandomizedSearchCV(
    minhash_hgbt,
    **random_search_params,
).fit(X_trans_train, y_train)

# %%
minhash_hgbt_tuned = TunedThresholdClassifierCV(
    minhash_hgbt_search.best_estimator_,
    **tuned_params,
).fit(
    X_trans_val,
    y_val,
    amount=X_val["total_price"],
)

results["minhash_hgbt_tuned"] = get_results(
    minhash_hgbt_tuned,
    X_trans_test,
    y_test,
    amount=X_test["total_price"],
    threshold=minhash_hgbt_tuned.best_threshold_,
)

plot_metric(results, metric_name="brier_score_loss")

# %%
plot_metric(results, metric_name="cost_gain")

# %%
# The aggregation scheme and minhash encoder improve the brier score, but the gains
# are very close.
#
# Let's now focus on the most important features for our model, using the
# ``permutation_importance`` function from scikit-learn.

from utils_modeling import plot_permutation_importance

n_subsample = 10_000  # makes the compute cheaper

plot_permutation_importance(
    minhash_hgbt_tuned,
    X_trans_test.head(n_subsample),
    y_test.head(n_subsample),
    random_state=0,
    figsize=(5, 9),
)

# %%
# Price statistics explain a lot of the target variance for our model, and some
# embedding dimensions too. MinHash embeddings aren't interpretable however, so we can't
# deduce which string token from the categories are actually important. Note that this
# could be achieved with the ``GapEncoder`` from skrub.
#
# NLP estimator
# -------------
#
# As a next model candidate, we look at semantic embeddings instead of syntaxic
# embeddings. This means using a naturel language encoder, that has been pretrained on a
# massive text dataset. The idea is by bringing more powerful embeddings to our final
# boosting tree estimator, we might see a boost in performance.
#
# We also change our aggregation scheme: instead of computing embedding then
# aggregating, we first aggregate categorical columns in a single prompt column, and
# then encode this prompt column using the language model. This will make computation
# cheaper, and help the encoder to better contextualize its inputs.
#
# The prompt follows the following structure:
#
#   `nbr1` model of `model1` made by `make1` in the category `item1`, and `nbr2` model
#   of `model2` ...
#
# Language models are generally less efficient with numbers, so we convert numbers into
# text: "1" becomes "one", "2" becomes "two" and so on with the ``inflect`` library.

import inflect


def make_prompt(X):
    number_converter = inflect.engine()

    prompt = pd.Series([""] * X.shape[0], index=X.index, name="prompt")

    for idx in range(MAX_ITEMS):
        # Add nbr
        nbr_col = get_group_cols("nbr")[idx]

        if idx > 0:
            nbr_col = get_group_cols("nbr")[idx]
            mask = X[nbr_col].notnull()
            prompt.loc[mask] += ", and "

        X_nbr = X[nbr_col].copy()
        mask = X_nbr.notnull()
        prompt.loc[mask] += (
            X_nbr.loc[mask]
            .astype(int)
            .apply(number_converter.number_to_words)
            .astype(str)
        ) + " model of "

        # Add model
        model_col = get_group_cols("model")[idx]
        X_model = X[model_col].copy()
        mask = X_model.notnull()
        prompt.loc[mask] += X_model.loc[mask] + " made by "

        # Add make
        make_col = get_group_cols("make")[idx]
        X_make = X[make_col].copy()
        mask = X_make.notnull()
        prompt.loc[mask] += X_make.loc[mask] + " in the category "

        # Add item
        item_col = get_group_cols("item")[idx]
        X_item = X[item_col].copy()
        mask = X_make.notnull()
        prompt.loc[mask] += X_item.loc[mask]

    return prompt.to_frame()


# %%
# Let's observe a few prompts.

make_prompt_transformer = FunctionTransformer(make_prompt)
prompts = make_prompt_transformer.fit_transform(X_train)
sample_prompts = prompts.iloc[9:14].reset_index(drop=True)
sample_prompts.values

# %%
# Next, we instantiate the encoder from the sentence transformer library, which
# downloads encoders from HuggingFace. Since we don't fine-tune it, this is a stateless
# operation.

from sentence_transformers import SentenceTransformer


def sentence_transformer_fn(X, **kwargs):
    original_index = X.index
    X = np.ravel(X)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    X_trans = model.encode(X, **kwargs)
    columns = get_group_cols("e", max_items=X_trans.shape[1])
    return pd.DataFrame(X_trans, columns=columns, index=original_index)


# %%
# We then run the sentence transformer on our 5 prompts above, and **observe the cosine
# similarity of the embeddings**.
#
# The second, third and fourth prompts contains respectively one iPhone and then one
# iPad each. The first and last prompt contains a Sony tv and a Baby chair. Therefore
# we expect the Apple product to have a similar embedding, and the rest to be very
# dissimilar.

sentence_transformer = FunctionTransformer(sentence_transformer_fn)
X_trans = sentence_transformer.fit_transform(sample_prompts)

# Equivalent to 1 - pairwise_distances(X_trans, metric="cosine")
similarity = X_trans @ X_trans.T
sns.heatmap(similarity, annot=True)

# %%
# This is indeed what we observe: the iPad similarities are at 0.83, the iPhone and
# iPads is 0.73, and the rest is below 0.3.
#
# How does ``MinHashEncoder`` embeddings compare? Note that **we have to use hamming
# distance with the Minhash**.

from sklearn.metrics import pairwise_distances

X_trans = MinHashEncoder().fit_transform(sample_prompts["prompt"])
similarity = 1 - pairwise_distances(X_trans, metric="hamming")
sns.heatmap(similarity, annot=True)

# %%
# The iPads embedding similarity has diminished, and the similarities between unrelated
# items has increased. Also, MinHash representations for iPhone and iPad are not
# similar. **So, it seems that our NLP encoder yields better representation than
# MinHash!**
#
# As the encoder dimensionality is quite high, let's truncate it using a PCA with
# 128 dimensions. The other transformers are the same as before, and we apply the same
# randomized search.

from sklearn.decomposition import PCA

nlp_encoder = Pipeline(
    [
        ("make_prompt", make_prompt_transformer),
        ("encoder", sentence_transformer),
        ("pca", PCA(n_components=128)),
    ]
)

vectorizer = FeatureUnion(
    [
        ("agg_number", row_aggregate_number_transformer),
        ("nlp_encoder", nlp_encoder),
        ("agg_goods_code", row_aggregate_target_encoder),
    ]
).set_output(transform="pandas")

vectorizer

# %%
X_trans_train = vectorizer.fit_transform(X_train, y_train)
X_trans_val = vectorizer.transform(X_val)
X_trans_test = vectorizer.transform(X_test)
X_trans_train


# %%
nlp_hgbt = HistGradientBoostingClassifier()

nlp_hgbt_search = RandomizedSearchCV(
    nlp_hgbt,
    **random_search_params,
).fit(X_trans_train, y_train)


# %%
nlp_hgbt_tuned = TunedThresholdClassifierCV(
    nlp_hgbt_search.best_estimator_,
    **tuned_params,
).fit(
    X_trans_val,
    y_val,
    amount=X_val["total_price"],
)

results["nlp_hgbt_tuned"] = get_results(
    nlp_hgbt_tuned,
    X_trans_test,
    y_test,
    amount=X_test["total_price"],
    threshold=nlp_hgbt_tuned.best_threshold_,
)

plot_metric(results, metric_name="brier_score_loss")


# %%
# The brier score between NLP and Minhash encoders are very close.

plot_metric(results, metric_name="cost_gain")

# %%
# The gains are very close too. So, the improvements brought by the NLP encoder is
# limited, although its embedding are more meaningful than MinHash embeddings.

# %%
from utils_modeling import plot_permutation_importance

n_subsample = 10_000

plot_permutation_importance(
    nlp_hgbt_tuned,
    X_trans_test.head(n_subsample),
    y_test.head(n_subsample),
    figsize=(5, 12),
)

# %%
# The permutation importances don't bring immediate insight, and we would need to
# inspect the partial dependencies of different embedding dimensions to get more
# understanding, before inspecting the prompts of the samples with the highest
# activations for these embeddings.
#
# Disagreeing models
# ------------------
#
# We inspect our two last models by first displaying their confusion matrices.

from utils_modeling import plot_confusion_matrix

plot_confusion_matrix(
    results,
    model_names=["minhash_hgbt_tuned", "nlp_hgbt_tuned"],
)

# %%
# Both models have comparable number of false negative.
#
# **Can we compare where these two models disagree?**


def get_disagreeing_between(model_name_1, model_name_2, X_test, y_test, results):
    y_pred_1 = get_y_pred_classes(results, model_name_1)
    y_pred_2 = get_y_pred_classes(results, model_name_2)

    mask = y_pred_1 != y_pred_2

    vectorizer = FeatureUnion(
        [
            ("agg_number", row_aggregate_number_transformer),
            ("nlp_encoder", make_prompt_transformer),
        ]
    ).set_output(transform="pandas")
    prompts = vectorizer.fit_transform(X_test)[["prompt", "cash_price_sum", "nbr_mean"]]

    disagreeing_prompts = prompts.loc[mask].copy()
    disagreeing_prompts[model_name_1] = y_pred_1[mask]
    disagreeing_prompts[model_name_2] = y_pred_2[mask]
    disagreeing_prompts["y_test"] = y_test.loc[mask]

    return disagreeing_prompts


def get_y_pred_classes(results, model_name):
    y_proba = results[model_name]["y_proba"]
    threshold = results[model_name]["model"].best_threshold_
    return (y_proba[:, 1] > threshold).astype("int32")


# %%
# We first display the prompts where the nlp model correctly predicts 1 (a fraud),
# while the minhash predicts 0 (a legit basket). We focus on false negative,
# since these are the most expensive errors.

# %%
disagreeing_prompts = get_disagreeing_between(
    "minhash_hgbt_tuned", "nlp_hgbt_tuned", X_test, y_test, results
)

pd.set_option("display.max_colwidth", 1000)
disagreeing_prompts.query("nlp_hgbt_tuned == 1 & y_test == 1")

# %%
# Then, we observe prompts where the minhash model correctly predicts 1 and the nlp
# model predicts 0.

disagreeing_prompts.query("minhash_hgbt_tuned == 1 & y_test == 1")

# %%
# The difference of the false negative between these two models is not obvious. Most
# disagreeing false negative baskets contains Apple products.
#
# Finally, let's observe prompts where both models yield false negative.


def get_agreeing_between(model_name_1, model_name_2, X_test, y_test, results):
    y_pred_1 = get_y_pred_classes(results, model_name_1)
    y_pred_2 = get_y_pred_classes(results, model_name_2)

    mask = y_pred_1 == y_pred_2

    vectorizer = FeatureUnion(
        [
            ("agg_number", row_aggregate_number_transformer),
            ("nlp_encoder", make_prompt_transformer),
        ]
    ).set_output(transform="pandas")
    prompts = vectorizer.fit_transform(X_test)[["prompt", "cash_price_sum", "nbr_mean"]]

    agreeing_prompts = prompts.loc[mask].copy()
    agreeing_prompts[model_name_1] = y_pred_1[mask]
    agreeing_prompts[model_name_2] = y_pred_2[mask]
    agreeing_prompts["y_test"] = y_test.loc[mask]

    return agreeing_prompts


# %%
agreeing_prompts = get_agreeing_between(
    "minhash_hgbt_tuned", "nlp_hgbt_tuned", X_test, y_test, results
)

agreeing_prompts.query("nlp_hgbt_tuned == 0 & y_test == 1").head(30)

# %%
# Interestingly, both models fail at detecting fraudulent basket containing a single
# Apple product. Maybe these baskets don't exhibit any difference with non fraudulent
# ones?
#
# Conclusion
# ----------
#
# We have compared three models against baselines, and seen a difference between our
# low effort against the Minhash aggregation, however NLP embeddings didn't show
# tangible improvement over the MinHash ones.
#
# This is understandable as the prompts were made over short textual categories with
# morphological differences with very little context available, a favourable context for
# a computationally cheap encoder like MinHash.
#
# We conclude this example with ROC curves, PR curves and the calibration curves of our
# models.

from utils_modeling import plot_roc_curve

plot_roc_curve(results)

# %%
# Minhash has the best ROC AUC.

from utils_modeling import plot_pr_curve

plot_pr_curve(results)

# %%
# MinHash has the best average precision (PR AUC), which is the initial metric of the
# challenge.

from utils_modeling import plot_calibration_curve

plot_calibration_curve(results)

# %%
# All calibration curves are extremely bad past the 40% predicted probability bin.
# The NLP encoder and MinHash models are slightly better calibrated than the minimal
# effort at the beginning. This disparity is due to the very low prevalence of the
# target: there is almost no prediction beyond 40%.
#
# If we wanted to use these predicted scores as confidence estimate, a good first step
# would be to calibrate our models using an Isotonic Regressor. This would reduce the
# calibration error **in average**, but wouldn't eliminate the **grouping loss**, i.e.
# the variance in calibration error within each bins, which would need to be examined
# separately.
#
# To go beyond
# ------------
#
# There are different directions we can think of to improve our modeling:
#
# - Adding confidence intervals on our metrics by using an outer cross validation on the
#   initial train, val and test splitting operation, via multiple random states.
# - Stacking a kNearestNeighbors transformer to aggregate the target, total_prices and
#   numeric columns, after vectorizing the categorical entities.
# - More expensive hyper-parameters tuning, in particular for the vectorizers output
#   dimensions.
# - Ensembling estimator with different inductive biases (kNN classifiers, linear
#   models)
# - Using a graph neural network classifier like CARTE, to jointly encode numeric and
#   categorical features representations.
