import polars as pl

from skrub._discover import Discover

if __name__ == "__main__":
    # working with binary to debug
    data_lake_path = "data/binary_update/*.parquet"
    base_table_path = "data/source_tables/yadl/movies_large-yadl-depleted.parquet"
    query_column = "col_to_embed"

    base_table = pl.read_parquet(base_table_path)

    discover = Discover(data_lake_path, [query_column])
    print("fitting")
    discover.fit(base_table)
    print("transforming")
    joined_table = discover.transform(base_table)
    print(joined_table)
