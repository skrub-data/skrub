import pandas as pd
import os
import requests
import pandas
import io

FOLDER_PATH = os.path.dirname(os.path.realpath(__file__))


def get_checksum(dataset_name):
    """
    function that will retrieve the checksum of a specific dataset
    from a dictionnary.
    :param dataset_name:
    :return:
    """
    return 0


def create_checksum(data):
    return 0


def verify_checksum(data, dataset_name):
    orig_checksum = get_checksum(dataset_name)
    data_checksum = create_checksum(data)
    assert orig_checksum == data_checksum, "error; a change has been detected in the data"


def _download_from_url(url, dataset_name=None):
    if dataset_name in ['employee_salaries']:
        return pd.read_csv(url)
    else:
        result = requests.get(url).content
        result = pd.read_csv(io.StringIO(result.decode('utf-8')))
    return result


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


def clean_employee_salaries(df):
    """
    data is already clean for employee_salaries, so no need to do anything
    :param df:
    :return:
    """
    return df


def read_or_download(url, local_path, local_only):
    """
    function that either reads or downloads the data, given an url and a local path
    :param url:
    :param local_path:
    :return:
    """
    if os.path.exists(local_path):
        data = pd.read_csv(local_path)
    else:
        if local_only:
            raise ValueError('no data was found locally')
        else:
            data = _download_from_url(url, dataset_name='employee_salaries')
    return data


def fetch_employee_salaries(local_only=False):
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

    url = "https://data.montgomerycountymd.gov/api/views/xj3h-s2i7/rows.csv?accessType=DOWNLOAD"
    local_path = 'data/employee_salaries.csv.gz'
    data = read_or_download(url, local_path, local_only)
    verify_checksum(data, 'employee_salaries')
    data.to_csv(local_path, compression='gzip')
    data = clean_employee_salaries(data)
    return data


if __name__ == '__main__':
    data = fetch_employee_salaries()
    print(data.head())
