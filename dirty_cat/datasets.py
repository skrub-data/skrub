import os


FOLDER_PATH = os.path.dirname(os.path.realpath(__file__))


def fetch_midwest_survey():
    """
    Function that returns the path to "midwest survey" example dataset.

    Returns
    ========

    filename: string
        The absolute path to the CSV file

    Notes
    =====

    The original data is retrieved from:
    https://github.com/fivethirtyeight/data/tree/master/region-survey
    """

    return os.path.join(FOLDER_PATH, 'data', 'midwest_survey.csv.gz')


def fetch_employee_salaries():
    '''
    Function that returns the path to "employee_salaries" example dataset.

    Returns
    ========

    filename: string
        The absolute path to the CSV file

    Notes
    =====

    The original data is retrieved from:
    https://catalog.data.gov/dataset/employee-salaries-2016
    '''

    return os.path.join(FOLDER_PATH, 'data', 'employee_salaries.csv.gz')
