"""
Datasets module of dirty cat
"""

from .fetching import get_data_dir
from .fetching import fetch_medical_charge
from .fetching import fetch_midwest_survey
from .fetching import fetch_employee_salaries
from .fetching import fetch_road_safety
from .fetching import fetch_open_payments
from .fetching import fetch_drug_directory
from .fetching import fetch_traffic_violations

__all__ = [
    'get_data_dir',
    'fetch_medical_charge',
    'fetch_midwest_survey',
    'fetch_employee_salaries',
    'fetch_road_safety',
    'fetch_open_payments',
    'fetch_drug_directory',
    'fetch_traffic_violations'
]
