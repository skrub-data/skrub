
#%%
%load_ext autoreload
%autoreload 2
#%%
import polars as pl
import pandas as pd
from skrub import _dropnull
import numpy as np

# %%
df_pl = pl.DataFrame(
    {"idx": [1,2,3], "value": [np.nan, np.nan, np.nan]}
)
df_pd = pd.DataFrame(
    {"idx": [1,2,3], "value": [np.nan, np.nan, np.nan]}
)

# %%
dn = _dropnull.DropNullColumn()
# %%
dn.fit_transform(df_pl["value"])

# %%
dn.fit_transform(df_pl["idx"])

# %%
dn.fit_transform(df_pd["value"])
# %%

from skrub import TableVectorizer
# %%
tv = TableVectorizer(drop_null_columns=True)


# %%
tv.fit_transform(df_pl)

# %%
