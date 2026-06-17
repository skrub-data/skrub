"""
Datasets: fetching real-world data and generating synthetic data.
=================================================================

Skrub bundles ready-to-use datasets (fetched from the web on first use)
and synthetic generators for examples and experimentation.

**Fetching** functions download real-world datasets for regression or
classification:

- :func:`fetch_bike_sharing`, :func:`fetch_california_housing`,
  :func:`fetch_country_happiness`, :func:`fetch_credit_fraud`,
  :func:`fetch_drug_directory`, :func:`fetch_employee_salaries`,
  :func:`fetch_flight_delays`, :func:`fetch_medical_charge`,
  :func:`fetch_midwest_survey`, :func:`fetch_movielens`,
  :func:`fetch_open_payments`, :func:`fetch_toxicity`,
  :func:`fetch_traffic_violations`, :func:`fetch_videogame_sales`
- :func:`get_data_dir` — path to the local cache directory.

**Generating** helpers create small synthetic dataframes for quick demos:

- :func:`make_deduplication_data`, :func:`toy_cities`,
  :func:`toy_orders`, :func:`toy_products`

Anything not listed in ``__all__`` is private and should not be used
directly.
"""

from ._fetching import (
    fetch_bike_sharing,
    fetch_california_housing,
    fetch_country_happiness,
    fetch_credit_fraud,
    fetch_drug_directory,
    fetch_employee_salaries,
    fetch_flight_delays,
    fetch_medical_charge,
    fetch_midwest_survey,
    fetch_movielens,
    fetch_open_payments,
    fetch_toxicity,
    fetch_traffic_violations,
    fetch_videogame_sales,
)
from ._generating import make_deduplication_data, toy_cities, toy_orders, toy_products
from ._utils import get_data_dir

__all__ = [
    "fetch_bike_sharing",
    "fetch_california_housing",
    "fetch_country_happiness",
    "fetch_credit_fraud",
    "fetch_drug_directory",
    "fetch_employee_salaries",
    "fetch_flight_delays",
    "fetch_medical_charge",
    "fetch_midwest_survey",
    "fetch_movielens",
    "fetch_open_payments",
    "fetch_toxicity",
    "fetch_traffic_violations",
    "fetch_videogame_sales",
    "get_data_dir",
    "make_deduplication_data",
    "toy_orders",
    "toy_products",
    "toy_cities",
]
