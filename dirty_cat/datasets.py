import os


FOLDER_PATH = os.path.dirname(os.path.realpath(__file__))


def fetch_midwest_survey():
    """
    function that fetches example data
    - either from the internet
    - or locally
    because the base example dataset (midwest survey) chosen is very small,
    it is locally stored in data/midwest_survey.csv
    in the future, for bigger dataset, this function will get the data
    from a specific url, and store it (optional) before returning it.
    :return:
    """

    return os.path.join(FOLDER_PATH, 'data', 'midwest_survey.csv.gz')


def fetch_employee_salaries():
    '''
    Fetch employee_salaries dataset
    '''

    return os.path.join(FOLDER_PATH, 'data', 'employee_salaries.csv.gz')
