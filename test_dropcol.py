
#%%
# %load_ext autoreload
# %autoreload 2
#%%
import polars as pl
import pandas as pd
from skrub import _dropnull

# %%
df_pl = pl.DataFrame(
    {"idx": [1,2,3], "value": [None, None, None]}
)
df_pd = df_pl.to_pandas()

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
tv = TableVectorizer(drop_null_columns=False)

# %%
tv.fit_transform(df_pl)
# %%
