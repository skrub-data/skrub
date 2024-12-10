# %%
import polars as pl

from skrub._discover import Discover

# %%
# if __name__ == "__main__":
# working with binary to debug
data_lake_path = "_scratch/ctu_financial/test/*.parquet"
base_table_path = "_scratch/ctu_financial/Financial_std_acc.parquet"
query_column = "account_id"

base_table = pl.read_parquet(base_table_path)

discover = Discover(data_lake_path, [query_column])
print("fitting")
discover.fit(base_table)
print("transforming")
ranking = discover.transform(base_table)
print(ranking)

# %%
df2 = pl.read_parquet("_scratch/ctu_financial/test/Financial_std_Loan_Acc.parquet")
df_ = base_table.join(df2, on="account_id")
# %%
base_table = pl.read_parquet(base_table_path)

discover = Discover(data_lake_path, ["loan_id"])
print("fitting")
discover.fit(df_)
print("transforming")
ranking = discover.transform(df_)
print(ranking)

# %%
