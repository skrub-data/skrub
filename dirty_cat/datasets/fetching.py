"""
fetching function to retrieve example dataset, using nilearn
fetching convention.

The parts of the nilearn fetching utils that have an obvious
meaning are directly copied. The rest is annoted.
"""
# -*- coding: utf-8 -*-
import os
import requests
import shutil
import urllib
from collections import namedtuple
import contextlib
import warnings

from ..datasets.utils import md5_hash, _check_if_exists, \
    _uncompress_file, \
    _md5_sum_file, get_data_dir

# in nilearn, urllib is used. Here the request package will be used
# trying to use requests as much as possible (everything except the
# parsing function)

# current data has been pulled from bigml, which require authentication
# for downloading. So for now we download the data from github
# however, the data differ a little bit from the two sources
# so we either have to implement login into _fetch_data
# or to reverse-engineer the processing script that can transform the data
# from git
# to the data from bigml
# this is true for bigml and midwest survey


DatasetInfo = namedtuple('DatasetInfo',
                         ['name', 'urlinfos', 'main_file', 'source'])
# a DatasetInfo Object is basically a tuple of UrlInfos object
# an UrlInfo object is composed of an url and the filenames contained
# in the request content
UrlInfo = namedtuple(
    'UrlInfo', ['url', 'filenames', 'uncompress', 'encoding'])

ROAD_SAFETY_CONFIG = DatasetInfo(
    name='road_safety',
    urlinfos=(
        UrlInfo(
            url="http://data.dft.gov.uk/road-accidents-safety-data/"
                "RoadSafetyData_2015.zip",
            filenames=(
                "Casualties_2015.csv",
                "Vehicles_2015.csv",
                "Accidents_2015.csv"
            ),
            uncompress=True, encoding='utf-8'),
        UrlInfo(
            url="http://data.dft.gov.uk/road-accidents-safety-data/"
                "MakeModel2015.zip",
            filenames=("2015_Make_Model.csv",),
            uncompress=True, encoding='utf-8'
        )
    ),
    main_file="Accidents_2015.csv",  # for consistency, all files are relevant,
    source="https://data.gov.uk/dataset/road-accidents-safety-data"
)

OPEN_PAYMENTS_CONFIG = DatasetInfo(
    name='open_payments',
    urlinfos=
    (
        UrlInfo(
            url='http://download.cms.gov/openpayments/PGYR13_P011718.ZIP',
            filenames=None, uncompress=True, encoding='utf-8'
        ),
    ),
    main_file='OP_DTL_GNRL_PGYR2013_P01172018.csv',  # same
    source='https://openpaymentsdata.cms.gov'
)

MIDWEST_SURVEY_CONFIG = DatasetInfo(
    name='midwest_survey',
    urlinfos=(
        UrlInfo(
            url="https://github.com/fivethirtyeight/data/tree/"
                "master/region-survey/FiveThirtyEight_Midwest_Survey.csv",
            filenames=(
                "FiveThirtyEight_Midwest_Survey.csv",
            ), uncompress=False, encoding='utf-8'
        ),
    ),
    main_file="FiveThirtyEight_Midwest_Survey.csv",
    source="https://github.com/fivethirtyeight/data/tree/ master/region-survey"
)
MEDICAL_CHARGE_CONFIG = DatasetInfo(
    name='medical_charge',
    urlinfos=(
        UrlInfo(
            url="https://www.cms.gov/Research-Statistics-Data-and-Systems/"
                "Statistics-Trends-and-Reports/Medicare-Provider-Charge-Data/"
                "Downloads/Inpatient_Data_2011_CSV.zip",
            filenames=(
                "Medicare_Provider_Charge_Inpatient_DRG100_FY2011.csv",
            ),
            uncompress=True, encoding='utf-8'

        ),
    ),
    main_file="Medicare_Provider_Charge_Inpatient_DRG100_FY2011.csv",
    source="https://www.cms.gov/Research-Statistics-Data-and-Systems/"
           "Statistics-Trends-and-Reports/Medicare-Provider-Charge-Data"
           "/Inpatient.html"
)

EMPLOYEE_SALARIES_CONFIG = DatasetInfo(
    name='employee_salaries',
    urlinfos=(
        UrlInfo(
            url="https://data.montgomerycountymd.gov/api/views/"
                "xj3h-s2i7/rows.csv?accessType=DOWNLOAD",
            filenames=("rows.csv",),
            uncompress=False, encoding='utf-8'
        ),
    ),
    main_file="rows.csv",
    source="https://catalog.data.gov/dataset/ employee-salaries-2016"
)

TRAFFIC_VIOLATIONS_CONFIG = DatasetInfo(
    name='traffic_violations',
    urlinfos=(
        UrlInfo(
            url="https://data.montgomerycountymd.gov/api/views/"
                "4mse-ku6q/rows.csv?accessType=DOWNLOAD",
            filenames=(
                "rows.csv",
            ), uncompress=False, encoding='utf-8'
        ),
    ),
    main_file="rows.csv",
    source="https://catalog.data.gov/dataset/ traffic-violations-56dda"
)

DRUG_DIRECTORY_CONFIG = DatasetInfo(
    name='drug_directory',
    urlinfos=(
        UrlInfo(
            url="https://www.accessdata.fda.gov/cder/ndctext.zip",
            filenames=(
                "product.txt",
                "package.txt",
            ), uncompress=True, encoding='latin-1'
        ),
    ),
    main_file="product.txt",
    source="https://www.fda.gov/Drugs/InformationOnDrugs/ucm142438.htm"
)

FOLDER_PATH = os.path.dirname(os.path.realpath(__file__))


class FileChangedError(Exception):
    pass


def _change_file_encoding(file_name, initial_encoding, target_encoding):

    temp_file_name = file_name + '.temp'
    try:
        with open(file_name, "r", encoding=initial_encoding) as source:
            with open(temp_file_name, "w", encoding=target_encoding) as target:
                while True:
                    contents = source.read(100000)
                    if not contents:
                        break
                    target.write(contents)
        shutil.move(temp_file_name, file_name)

    finally:
        if os.path.exists(temp_file_name):
            os.unlink(temp_file_name)


def _download_and_write(url, file, show_progress=True):
    if show_progress:  # maybe not an ideal design, should we mark
        # clint as mandatory?
        from clint.textui import progress
    try:
        # using stream=True to download the response body only when
        # accessing the content attribute
        from ..datasets.utils import request_get
        with contextlib.closing(request_get(url, stream=True)) as r:
            total_length = r.headers.get('Content-Length')
            if total_length is not None:
                with open(file, 'wb') as local_file:
                    content_iterator = r.iter_content(chunk_size=1024)
                    if show_progress:
                        content_iterator = progress.bar(
                            content_iterator, expected_size=(int(total_length) /
                                                             1024) + 1)

                    for chunk in content_iterator:
                        if chunk:
                            local_file.write(chunk)
                            local_file.flush()

            else:
                warnings.warn('content size cannot be found, '
                              'downloading file from {} as a whole'.format(
                    url))
                with open(file, 'wb') as local_file:
                    local_file.write(r.content)

    except requests.RequestException as e:
        # pretty general request exception. subject to change
        raise Exception('error while fetching: {}'.format(e))


def fetch_dataset(configfile: DatasetInfo, show_progress=True):
    data_dir = os.path.join(get_data_dir(), configfile.name)
    for urlinfo in configfile.urlinfos:
        _fetch_file(urlinfo.url, data_dir, filenames=urlinfo.filenames,
                    uncompress=urlinfo.uncompress, show_progress=show_progress,
                    initial_encoding=urlinfo.encoding)
    # returns the absolute path of the csv file where the data is
    result_dict = {
        'description': 'The downloaded data contains the {} dataset.\n'
                       'It can originally be found at: {}'.format(
            configfile.name, configfile.source),
        'path': os.path.join(data_dir, configfile.main_file)
    }
    return result_dict


def _fetch_file(url, data_dir, filenames=None, overwrite=False,
                md5sum=None, uncompress=True, show_progress=True,
                initial_encoding='utf-8'):
    """fetches the content of a requested url

    IF the downloaded file is compressed, then the fetcher
    looks also for the uncompressed files before downloading .


    Parameters
    ----------
    url: str
        url from where to fetch the file from
    data_dir: str
        directory where the data will be stored
    filenames: list
        names of the files in the url content
    overwrite: bool
        whether to overwrite present data
    md5sum: str
        if provided, verifies the integrity of the file using a hash
    uncompress: bool
        whether to uncompress the content of the url

    show_progress:
        if ``True``, displays a progressbar during the downloading of the
        dataset. Warning: ``clint`` needs to be implemented and is not in the
        requirements for now

    Returns
    -------
    a dictionary containing:

        - a short description of the dataset (under the ``description`` key )
        - an absolute path leading to the csv file where the data is stored
          locally (under the ``path`` key)

    NOTES
    -----
    NON-implemented nilearn parameters:
    * the ``resume`` option, that would resume partially downloaded files
    * username/password
    * handlers

    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Determine filename using URL. sticking to urllib.parse, requests does not
    # provide parsing tools
    parse = urllib.parse.urlparse(url)
    file_name = os.path.basename(parse.path)
    download = False
    if file_name == '':
        file_name = md5_hash(parse.path)

    temp_file_name = file_name + ".part"
    full_name = os.path.join(data_dir, file_name)
    temp_full_name = os.path.join(data_dir, temp_file_name)
    if overwrite:
        download = True
        files_to_overwrite = [file_name, temp_file_name]
        if filenames:
            files_to_overwrite += filenames
        for name in files_to_overwrite:
            # remove all compressed/uncompressed files
            _check_if_exists(os.path.join(data_dir, name), remove=True)
    else:
        # if no filenames info provided, overwrite
        is_file_missing = True
        if filenames is not None:
            # first look for uncompressed files
            is_file_missing = any(
                [not _check_if_exists(os.path.join(data_dir, name),
                                      remove=False)
                 for name in filenames])
        if is_file_missing:
            # then look for compressed files
            if not _check_if_exists(full_name, remove=False):
                download = True

    if download:
        _download_and_write(url, temp_full_name, show_progress=show_progress)

    # chunk writing is not implemented, see if necessary
    if _check_if_exists(temp_full_name, remove=False):
        if md5sum is not None:
            if _md5_sum_file(temp_full_name) != md5sum:
                raise FileChangedError(
                    "File %s checksum verification has failed."
                    "Dataset fetching aborted." % temp_full_name)

        shutil.move(temp_full_name, full_name)
    if _check_if_exists(full_name, remove=False) and uncompress:
        _uncompress_file(full_name, delete_archive=True)

    if download and (initial_encoding != 'utf-8'):
        for file in os.listdir(data_dir):
            _change_file_encoding(
                os.path.join(data_dir, file), initial_encoding, 'utf-8')
    return full_name


def fetch_employee_salaries():
    """fetches the employee_salaries dataset

    The employee_salaries dataset contains information about annual salaries
    (year 2016) for more than 9,000 employees of the Montgomery County
    (Maryland, US).


    Returns
    -------
    dict
        a dictionary containing:

            - a short description of the dataset (under the ``description``
              key)
            - an absolute path leading to the csv file where the data is stored
              locally (under the ``path`` key)

    References
    ----------
    https://catalog.data.gov/dataset/employee-salaries-2016

    """

    return fetch_dataset(EMPLOYEE_SALARIES_CONFIG, show_progress=False)


def fetch_road_safety():
    """fetches the road safety dataset

    Returns
    -------
    dict
        a dictionary containing:

            - a short description of the dataset (under the ``description``
              key)
            - an absolute path leading to the csv file where the data is stored
              locally (under the ``path`` key)


    References
    ----------
    https://data.gov.uk/dataset/road-accidents-safety-dataset
    """

    return fetch_dataset(ROAD_SAFETY_CONFIG, show_progress=False)


def fetch_medical_charge():
    """fetches the medical charge dataset

    Returns
    -------
    dict
        a dictionary containing:

            - a short description of the dataset (under the ``description``
              key)
            - an absolute path leading to the csv file where the data is stored
              locally (under the ``path`` key)


    References
    ----------
    https://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Medicare-Provider-Charge-Data/Inpatient.html"
    """
    return fetch_dataset(MEDICAL_CHARGE_CONFIG, show_progress=False)


def fetch_midwest_survey():
    """fetches the midwest survey dataset

    Returns
    -------
    dict
        a dictionary containing:

            - a short description of the dataset (under the ``description``
              key)
            - an absolute path leading to the csv file where the data is stored
              locally (under the ``path`` key)


    References
    ----------
    https://github.com/fivethirtyeight/data/tree/master/region-survey
    """
    return fetch_dataset(MIDWEST_SURVEY_CONFIG, show_progress=False)


def fetch_open_payments():
    """fetches the open payements dataset

    Returns
    -------
    dict
        a dictionary containing:

            - a short description of the dataset (under the ``description``
              key)
            - an absolute path leading to the csv file where the data is stored
              locally (under the ``path`` key)


    References
    ----------
    https://openpaymentsdata.cms.gov
    """
    return fetch_dataset(OPEN_PAYMENTS_CONFIG, show_progress=False)


def fetch_traffic_violations():
    """fetches the traffic violations dataset

    Returns
    -------
    dict
        a dictionary containing:

            - a short description of the dataset (under the ``description``
              key)
            - an absolute path leading to the csv file where the data is stored
              locally (under the ``path`` key)


    References
    ----------
    https://catalog.data.gov/dataset/traffic-violations-56dda
    """
    return fetch_dataset(TRAFFIC_VIOLATIONS_CONFIG, show_progress=False)


def fetch_drug_directory():
    """fetches the drug directory dataset

    Returns
    -------
    dict
        a dictionary containing:

            - a short description of the dataset (under the ``description``
              key)
            - an absolute path leading to the csv file where the data is stored
              locally (under the ``path`` key)


    References
    ----------
    https://www.fda.gov/Drugs/InformationOnDrugs/ucm142438.htm
    """
    return fetch_dataset(DRUG_DIRECTORY_CONFIG, show_progress=False)
