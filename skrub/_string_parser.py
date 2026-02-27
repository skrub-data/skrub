from . import _dataframe as sbd
from ._on_each_column import SingleColumnTransformer


class StringParser(SingleColumnTransformer):
    """
    """
    def __init__(self, dictionary):
        self.dictionary = dictionary
    
    def fit_transform(self, X, y=None):
        del y

        return self.transform(X)

    def transform(self, X):
        breakpoint()
        col_name = sbd.name(X) or "string"

        X_str = X.astype(str).str.lower().str.strip()

        pattern = (
            r"(?:(?P<unit1>[a-zA-Z]+)\s*(?P<value1>[-+]?\d*\.?)"
            r"|(?:(?P<value2>[-+]?\d*\.?\d+)\s*(?P<unit2>[a-zA-Z]+))")
        
        parsed = X_str.str.extract(pattern)
        parsed["unit"] = parsed["unit1"].combine_first(parsed["unit2"])
        parsed["value"] = parsed["value1"].combine_first(parsed["value2"])
        
        parsed = parsed.drop(columns=["unit1", "value1", "unit2", "value2"])

        parsed["value"] = parsed["value"].str.replace(",", ".").astype(float)

        base_unit = next(
            (u for u, f in self.dictionary.items() if f == 1.0),
            list(self.dictionary.keys())[0],
        )

        parsed["unit"] = parsed["unit"].fillna(base_unit)

        parsed["factor"] = parsed["unit"].map(self.dictionary)
        result = parsed["value"] * parsed["factor"]
        result.name = f"{col_name}_{base_unit}"
        result.index = X.index

        self.input_name_ = col_name
        self.all_outputs_ = [result.name]
        self._is_fitted = True

        return result

