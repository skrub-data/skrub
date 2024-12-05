# %% test string encoder
import polars as pl
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from skrub._string_encoder import StringEncoder

corpus = [
    "this is the first document",
    "this document is the second document",
    "and this is the third one",
    "is this the first document",
]
column = pl.Series(name="this_column", values=corpus)

# %%

pipe = Pipeline(
    [
        ("tfidf", TfidfVectorizer()),
        ("pca", PCA(n_components=2)),
    ]
)
# %%
a = pipe.fit_transform(corpus)

# %%
se = StringEncoder(2)

# %%
r = se.fit_transform(column)
# %%
