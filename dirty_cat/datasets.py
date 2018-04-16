"""
fetching function to retrieve example dataset, using nilearn
fetching convention.

The parts of the nilearn fetching utils that have an obvious
meaning are directly copied. The rest is annoted.
"""

import shutil
import pandas as pd
import os
import requests
import pandas
import io
import hashlib
import urllib
from clint.textui import \
    progress  # package developed by the same guyy than requests
import warnings
import zipfile
from collections import namedtuple

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
                         ['name', 'urlinfos'])

# a DatasetInfo Object is basically a tuple of UrlInfos object
# an UrlInfo object is composed of an url and the filenames contained
# in the request content
UrlInfo = namedtuple('UrlInfo', ['url', 'filenames', 'uncompress'])

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
            uncompress=True),
        UrlInfo(
            url="http://data.dft.gov.uk/road-accidents-safety-data/"
                "MakeModel2015.zip",
            filenames=("2015_Make_Model.csv",),
            uncompress=True
        )
    )
)

OPEN_PAYMENTS_CONFIG = DatasetInfo(
    name='open_payments',
    urlinfos=
    (
        UrlInfo(
            url='http://download.cms.gov/openpayments/PGYR13_P011718.ZIP',
            filenames=None, uncompress=True
        ),
    )
)

MIDWEST_SURVEY_CONFIG = DatasetInfo(
    name='midwest_survey',
    urlinfos=(
        UrlInfo(
            url="https://github.com/fivethirtyeight/data/tree/"
                "master/region-survey/FiveThirtyEight_Midwest_Survey.csv",
            filenames=(
                "FiveThirtyEight_Midwest_Survey.csv",
            ), uncompress=False
        ),
    )
)
MEDICAL_CHARGE_CONFIG = DatasetInfo(
    name='medical_charge',
    urlinfos=(
        UrlInfo(
            url="https://www.cms.gov/Research-Statistics-Data-and-Systems/"
                "Statistics-Trends-and-Reports/Medicare-Provider-Charge-Data/"
                "Downloads/Inpatient_Data_2011_CSV.zip",
            filenames=(
                "MedicalProviderChargeInpatient.csv",
            ),
            uncompress=True

        ),
    )
)

EMPLOYEE_SALARIES_CONFIG = DatasetInfo(
    name='employee_salaries',
    urlinfos=(
        UrlInfo(
            url="https://data.montgomerycountymd.gov/api/views/"
                "xj3h-s2i7/rows.csv?accessType=DOWNLOAD",
            filenames=("rows.csv",),
            uncompress=False
        ),
    )
)

TRAFFIC_VIOLATIONS_CONFIG = DatasetInfo(
    name='traffic_violations',
    urlinfos=(
        UrlInfo(
            url="https://data.montgomerycountymd.gov/api/views/"
                "4mse-ku6q/rows.csv?accessType=DOWNLOAD",
            filenames=(
                "rows.csv",
            ), uncompress=True
        ),
    )
)


def md5_hash(string):
    m = hashlib.md5()
    m.update(string)
    return m.hexdigest()


def _md5_sum_file(path):
    """ Calculates the MD5 sum of a file.
    """
    with open(path, 'rb') as f:
        m = hashlib.md5()
        while True:
            data = f.read(8192)
            if not data:
                break
            m.update(data)
    return m.hexdigest()


FOLDER_PATH = os.path.dirname(os.path.realpath(__file__))


def _uncompress_file(file_, delete_archive=True):
    """Uncompress files contained in a data_set.


    Parameters
    ----------
    file_: path to file
    delete_archive: whether to delete the compressed file afterwards


    Returns
    -------
    None if everything worked out fine
    ValueError otherwise


    Notes
    -----
    only supports zip and gzip
    """
    data_dir = os.path.dirname(file_)
    filename, ext = os.path.splitext(file_)

    if zipfile.is_zipfile(file_):
        z = zipfile.ZipFile(file_)
        z.extractall(path=data_dir)
        z.close()
        if delete_archive:
            os.remove(file_)
    elif ext == '.gz':
        import gzip
        gz = gzip.open(file_)
        if ext == '.tgz':
            filename = filename + '.tar'
        out = open(filename, 'wb')
        shutil.copyfileobj(gz, out, 8192)
        gz.close()
        out.close()
        # If file is .tar.gz, this will be handle in the next case
        if delete_archive:
            os.remove(file_)
    else:
        raise IOError('[Compression] unknown archive format {}'.format(ext))


def _check_if_exists(path, remove=False):
    if remove:
        try:
            os.remove(path)
        except OSError:
            pass
        return False
    else:
        return os.path.exists(path)


def fetch_dataset(configfile: DatasetInfo):
    data_dir = os.path.join(get_data_dir(), configfile.name)
    for urlinfo in configfile.urlinfos:
        _fetch_file(urlinfo.url, data_dir, filenames=urlinfo.filenames,
                    uncompress=urlinfo.uncompress)


def _fetch_file(url, data_dir, filenames=None, overwrite=False,
                md5sum=None, uncompress=True):
    """fetches the content of a requested url

    IF the downloaded file is compressed, then the fetcher
    looks also for the uncompressed files before downloading .


    Parameters
    ----------
    url
    data_dir: directory where the data will be stored
    filenames: names of the files in the url content
    overwrite: whether to overwrite present data
    md5sum: if provided, verifies the integrity of the file using a hash
    uncompress: whether to uncompress the content of the url

    Returns
    -------
    the full name of the extracted file

    NOTES
    -----
    NON-implemented nilearn parameters:
    * the resume option, that would resume partially downloaded files
    * username/password
    * handlers

    """
    # TODO: look for uncompressed files when download result is zippped.
    # potentially passing the config as an argument because this function
    # does a lot of thing, maybe makes sense to group the argument into
    # the config file
    # Determine data path
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
        for name in [file_name, temp_file_name, *filenames]:
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
        try:
            # using stream=True to download the response body only when
            # accessing the content attribute
            with requests.get(url, stream=True) as r:
                total_length = r.headers.get('Content-Length')
                if total_length is not None:
                    with open(temp_full_name, 'wb') as local_file:
                        for chunk in progress.bar(
                                r.iter_content(chunk_size=1024),
                                expected_size=(int(total_length) /
                                               1024) + 1):
                            if chunk:
                                local_file.write(chunk)
                                local_file.flush()
                else:
                    warnings.warn('content size cannot be found, '
                                  'downloading file from {} as a whole'.format(
                        url))
                    with open(temp_full_name, 'wb') as local_file:
                        local_file.write(r.content)

        except requests.RequestException as e:
            # pretty general request exception. subject to change
            raise Exception('error while fetching: {}'.format(e))

    # chunk writing is not implemented, see if necessary
    if _check_if_exists(temp_full_name, remove=False):
        if md5sum is not None:
            if (_md5_sum_file(temp_full_name) != md5sum):
                raise ValueError("File %s checksum verification has failed."
                                 "Dataset fetching aborted." % temp_full_name)

        shutil.move(temp_full_name, full_name)
    if _check_if_exists(full_name, remove=False) and uncompress:
        _uncompress_file(full_name, delete_archive=True)

    return full_name


def fetch_employee_salaries():
    return fetch_dataset(EMPLOYEE_SALARIES_CONFIG)


def fetch_road_safety():
    return fetch_dataset(ROAD_SAFETY_CONFIG)


def fetch_medical_charge():
    return fetch_dataset(MEDICAL_CHARGE_CONFIG)


def fetch_midwest_survey():
    return fetch_dataset(MIDWEST_SURVEY_CONFIG)


def fetch_open_payments():
    return fetch_dataset(OPEN_PAYMENTS_CONFIG)


def get_data_dir(name=None):
    """ Returns the directories in which nilearn looks for data.

    This is typically useful for the end-user to check where the data is
    downloaded and stored.

    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if name is not None:
        data_dir = os.path.join(data_dir, name)
    return data_dir


if __name__ == '__main__':
    fetch_midwest_survey()
    fetch_medical_charge()
    fetch_road_safety()
    fetch_employee_salaries()
    fetch_open_payments()
