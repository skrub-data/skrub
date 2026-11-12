# %%
import polars as pl

from skrub._discover import Discover, find_unique_values

# %%
# working with binary to debug
data_lake_path = "data/binary_update/*.parquet"
base_table_path = "data/source_tables/company_employees-yadl-depleted.parquet"
query_column = "col_to_embed"


base_table = pl.read_parquet(base_table_path)
# %%
find_unique_values(base_table, ["col_to_embed"])
# %%
discover = Discover(data_lake_path, [query_column])
print("fitting")
discover.fit(base_table)
print("transforming")
ranking = discover.transform(base_table)
print(ranking)

# %%
