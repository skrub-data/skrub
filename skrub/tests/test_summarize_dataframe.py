"""Minimal reproducer: does skrub crash on string columns containing '$$'?
Run with:
    uv run repro_skrub_dollar_sign.py
"""

import pandas as pd

from skrub._reporting._summarize import summarize_dataframe

df = pd.DataFrame(
    {
        "text": [
            "hello world",
            "foo bar",
            "this is not latex $$ just a double dolar sign",
        ]
    }
)
try:
    summarize_dataframe(df, with_plots=True, title=None, verbose=0)
    print("PASS: no error")
except ValueError as e:
    print(f"FAIL: {e}")
