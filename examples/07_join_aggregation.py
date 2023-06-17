"""
TODO : Add some docstring
----
"""
#!/usr/bin/env python
# coding: utf-8

# # Movies recommendation with explicit labels

# ## Loading data

# In[1]:


from skrub.datasets import fetch_movielens

ratings = fetch_movielens(dataset_id="ratings")
ratings = ratings.X
ratings


# In[2]:


import pandas as pd

ratings["timestamp"] = pd.to_datetime(ratings["timestamp"], unit="s")
ratings["year"] = ratings["timestamp"].dt.year
ratings["day_name"] = ratings["timestamp"].dt.day_name()


# ## Quick Data Exploration

# In[3]:


from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


def make_barplot(x, y, title):
    norm = plt.Normalize(y.min(), y.max())
    cmap = plt.get_cmap("magma")

    ax = sns.barplot(
        x=x,
        y=y,
        palette=cmap(norm(y)),
    )
    plt.xticks(rotation=30)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.tight_layout()
    plt.title(title)


# In[4]:


year_volume = ratings["year"].value_counts().sort_index()

make_barplot(
    x=year_volume.index,
    y=year_volume.values,
    title="Yearly volume of ratings",
)


# In[5]:


ratings["day_index"] = ratings["timestamp"].dt.weekday

day_name_index = (
    ratings[["day_index", "day_name"]].drop_duplicates().set_index("day_index")
)
day_name_index.sort_index()


# In[6]:


daily_volume = (
    ratings["day_index"].value_counts().sort_index().to_frame().join(day_name_index)
)

make_barplot(
    x=daily_volume["day_name"], y=daily_volume["count"], title="Daily volume of ratings"
)


# In[7]:


user_count_ratings = (
    ratings.groupby("userId").agg(n_ratings=("rating", "count")).reset_index()
)

print(
    "min:",
    user_count_ratings["n_ratings"].min(),
    "max:",
    user_count_ratings["n_ratings"].max(),
)

ax = sns.histplot(user_count_ratings["n_ratings"])
ax.set(
    xlabel="Number of ratings given by users",
    ylabel="Total number of users",
    title="Number of ratings given by user",
)


# In[8]:


rating_count = ratings["rating"].value_counts().sort_index()

make_barplot(
    x=rating_count.index,
    y=rating_count.values,
    title="Distribution of ratings given to movies",
)


# In[9]:


## Split train test by date

min_date = ratings["timestamp"].min()
max_date = ratings["timestamp"].max()
duration = max_date - min_date
train_threshold = max_date - duration / 3

train = ratings.loc[ratings["timestamp"] < train_threshold]
test = ratings.loc[ratings["timestamp"] >= train_threshold]
train.shape, test.shape


# ## Feature Engineering

# ***Ohe Hot Encoding day names***

# In[10]:


from sklearn.pipeline import make_pipeline, FunctionTransformer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

col_trans = make_column_transformer(
    ("passthrough", ["userId", "movieId", "rating"]),
    (OneHotEncoder(sparse_output=False), ["day_name"]),
    remainder="drop",
    verbose_feature_names_out=False,
)

train = col_trans.fit_transform(train)
train = pd.DataFrame(train, columns=col_trans.get_feature_names_out())
train.head()


# ***Join aggregator on `movieId` and `userId`***

# In[11]:


from skrub import JoinAggregator

cols_day_name = list(set(train.columns) - set(["userId", "movieId"]))

join_agg_rating = JoinAggregator(
    tables=[
        (train, "userId", cols_day_name),
        (train, "movieId", cols_day_name),
    ],
    main_key=["userId", "movieId"],
    suffixes=["_user", "_movie"],
    agg_ops=["sum", "mean"],
)
train = join_agg_rating.fit_transform(train[["userId", "movieId", "rating"]])

cols_to_drop = [
    col
    for col in train.columns
    if ("mean" in col and "day_name" in col) or ("rating_mean" in col)
]
train = train.drop(cols_to_drop, axis=1)
train


# ### Encoding movies

# In[12]:


movies = fetch_movielens(dataset_id="movies")
movies = movies.X
movies


# In[13]:


movies["genres"] = movies["genres"].str.replace("|", " ")
all_genres = movies["genres"].str.split().explode().unique()[:-3]
all_genres


# In[14]:


from sklearn.feature_extraction.text import CountVectorizer
from skrub import MinHashEncoder

vectorizer = CountVectorizer(
    vocabulary=all_genres,
    ngram_range=(1, 1),
    lowercase=False,
)

n_components = 10
min_hash_encoder = MinHashEncoder(n_components=n_components)

movie_transformer = make_column_transformer(
    ("passthrough", ["movieId"]),
    (min_hash_encoder, ["title"]),
    (vectorizer, "genres"),
    sparse_threshold=0,
)
movie_embedding = movie_transformer.fit_transform(movies)

embedding_cols = [f"title_x{idx}" for idx in range(n_components)] + all_genres.tolist()
movie_embedding = pd.DataFrame(movie_embedding, columns=["movieId", *embedding_cols])
movie_embedding


# ## Encoding users

# In[15]:


train_user_movie_embedding = train[["userId", "movieId"]].merge(
    movie_embedding, on="movieId", how="left"
)
train_user_movie_embedding.shape


# ## Join Aggregate user and movie embeddings!

# In[16]:


join_agg_embedding = JoinAggregator(
    tables=[
        (movie_embedding, "movieId", embedding_cols),
        (train_user_movie_embedding, "userId", embedding_cols),
    ],
    main_key=["movieId", "userId"],
    agg_ops=["mean"],
    suffixes=["_movie", "_user"],
)
train_embedding = join_agg_embedding.fit_transform(train)
train_embedding


# ## Putting everything together!

# In[17]:


def get_year_and_day_name(ratings):
    timestamps = pd.to_datetime(ratings["timestamp"], unit="s")
    day_names = timestamps.dt.day_name()
    return pd.get_dummies(day_names)


pipe_ohe_days = make_pipeline(
    FunctionTransformer(get_year_and_day_name),
)


# In[74]:


from sklearn.preprocessing import OneHotEncoder


def get_day_name(ratings):
    timestamps = pd.to_datetime(ratings["timestamp"], unit="s")
    day_names = timestamps.dt.day_name()
    return day_names.to_frame()


ft_get_day_name = FunctionTransformer(get_day_name)

pipe_ohe_day_name = make_pipeline(
    ft_get_day_name,
    OneHotEncoder(sparse_output=False, drop="if_binary"),
)

cols_base = ["userId", "movieId"]
pipe_ohe_day_name.fit(ratings)
cols_days = pipe_ohe_day_name[-1].get_feature_names_out().tolist()

ct_ohe_days = make_column_transformer(
    ("passthrough", cols_base),
    (pipe_ohe_day_name, ["timestamp"]),
    remainder="drop",
    # verbose_feature_names_out=False,
)

ft_to_pandas_days = FunctionTransformer(
    lambda X: pd.DataFrame(X, columns=cols_base + cols_days)
)

join_agg_rating = JoinAggregator(
    tables=[
        ("X", "userId", cols_days),
        ("X", "movieId", cols_days),
    ],
    main_key=["userId", "movieId"],
    suffixes=["_user", "_movie"],
    agg_ops=["sum"],
)

ft_filter_columns_days = FunctionTransformer(
    lambda df: df.drop(cols_base + cols_days, axis=1)
)

aggregation_days = make_pipeline(
    ct_ohe_days,
    ft_to_pandas_days,
    join_agg_rating,
    ft_filter_columns_days,
)
aggregation_days


# In[75]:


aggregation_days.fit_transform(ratings)


# In[76]:


def merge_movie_embedding(X):
    return X.merge(movie_embedding, on="movieId", how="left")


ft_merge_movie_embedding = FunctionTransformer(merge_movie_embedding)

ct_merge_movie_embedding = make_column_transformer(
    ("passthrough", ["userId"]),
    (ft_merge_movie_embedding, ["movieId"]),
    remainder="drop",
    verbose_feature_names_out=True,
)

cols_embedding = ft_merge_movie_embedding.fit_transform(
    ratings[["movieId"]]
).columns.tolist()

ft_to_pandas_embedding = FunctionTransformer(
    lambda X: pd.DataFrame(X, columns=["userId"] + cols_embedding)
)

join_agg_embedding = JoinAggregator(
    tables=[
        (movie_embedding, "movieId", cols_embedding),
        ("X", "userId", cols_embedding),
    ],
    main_key=["movieId", "userId"],
    agg_ops=["mean"],
    suffixes=["_movie", "_user"],
)

ft_filter_cols_embedding = FunctionTransformer(lambda df: df.drop(cols_base, axis=1))

aggregation_embedding = make_pipeline(
    ct_merge_movie_embedding,
    ft_to_pandas_embedding,
    join_agg_embedding,
    ft_filter_cols_embedding,
)
aggregation_embedding


# In[77]:


aggregation_embedding.fit_transform(ratings)


# In[78]:


from sklearn.compose import make_column_selector
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import HistGradientBoostingRegressor

feature_engineering = FeatureUnion(
    [
        ("aggregation_days", aggregation_days),
        ("aggregation_embedding", aggregation_embedding),
    ]
)

regressor_pipeline = make_pipeline(
    feature_engineering,
    HistGradientBoostingRegressor(),
)

regressor_pipeline


# ## Predict and score

# In[90]:


from sklearn.model_selection import TimeSeriesSplit, cross_validate

ratings = fetch_movielens(dataset_id="ratings").X
ratings = ratings.sort_values("timestamp").reset_index(drop=True)
X = ratings[["movieId", "userId", "timestamp"]]
y = ratings["rating"]

tscv = TimeSeriesSplit()
cross_validate(regressor_pipeline, X, y, cv=tscv)


# In[ ]:
