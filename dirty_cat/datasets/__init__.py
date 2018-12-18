"""
datasets module of dirty cat
"""

from .fetching import fetch_midwest_survey, fetch_employee_salaries, \
    fetch_open_payments, fetch_road_safety, fetch_medical_charge, \
    fetch_traffic_violations, fetch_drug_directory
from .fetching import get_data_dir

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
