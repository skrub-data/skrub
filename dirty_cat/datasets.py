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
# trying to use requests as much as possible (everything excpet the
# parsing function)

class FileFetcherConfig:
    def __init__(self, name, urls: (str, tuple), paths: (str, tuple),
                 opts=None):
        self.name = name
        self.urls = tuple(urls)
        self.paths = tuple(paths)
        self.opts = opts


# current data has been pulled from bigml, which require authentification
# for downlading. So for now we download the data from github
# however, the data differ a little bit from the two sources
# so we either have to implement login into _fetch_data
# or to reverse-engineer the processing script that can transform the data
# from git
# to the data from bigml
# this is true for bigml and midwest survey


FetcherConfig = namedtuple('FetcherConfig',
                           ['dataset_name', 'urlinfos'])

# a FetcherConfig Object is basically a tuple of UrlInfos object
# an UrlInfo object is composed of an url and the filenames containted
# in the request content
UrlInfo = namedtuple('UrlInfo', ['url', 'filenames', 'opts'])

road_safety_config = FetcherConfig(
    'road_safety',
    (
        UrlInfo(
            "http://data.dft.gov.uk/road-accidents-safety-data/"
            "RoadSafetyData_2015.zip",
            (
                "Casualties_2015.csv",
                "Vehicles_2015.csv",
                "Accidents_2015.csv"
            ),
            {"uncompress": True}
        ),
        UrlInfo(
            "http://data.dft.gov.uk/road-accidents-safety-data/"
            "MakeModel2015.zip",
            ("2015_Make_Model.csv",),
            {"uncompress": True}
        )
    )
)

open_payments_config = FetcherConfig(
    'open_payments',
    (
        UrlInfo(
            'http://download.cms.gov/openpayments/PGYR13_P011718.ZIP',
            None,
            {"uncompress": True}
        ),
    )
)

midwest_survey_config = FetcherConfig(
    'midwest_survey',
    (
        UrlInfo(
            "https://github.com/fivethirtyeight/data/tree/"
            "master/region-survey/FiveThirtyEight_Midwest_Survey.csv",
            (
                "FiveThirtyEight_Midwest_Survey.csv",
            ),
            {"uncompress": False}
        ),
    )
)
medical_charge_config = FetcherConfig(
    'medical_charge',
    (
        UrlInfo(
            "https://www.cms.gov/Research-Statistics-Data-and-Systems/"
            "Statistics-Trends-and-Reports/Medicare-Provider-Charge-Data/"
            "Downloads/Inpatient_Data_2011_CSV.zip",
            (
                "MedicalProviderChargeInpatient.csv",
            ),
            {'uncompress': True}
        ),
    )
)

employee_salaries_config = FetcherConfig(
    'employee_salaries',
    (UrlInfo(
        "https://data.montgomerycountymd.gov/api/views/"
        "xj3h-s2i7/rows.csv?accessType=DOWNLOAD",
        ("rows.csv",),
        {'uncompress': False}),))

traffic_violations_config = FetcherConfig(
    'traffic_violations',
    (
        UrlInfo(
            "https://data.montgomerycountymd.gov/api/views/"
            "4mse-ku6q/rows.csv?accessType=DOWNLOAD",
            (
                "rows.csv",
            ),
            {"uncompress": True}),

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
    """
    Uncompress files contained in a data_set.

    only supports zip and gzip
    :param file_:
    :param delete_archive:
    :return:
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


def fetch_dataset(configfile: FetcherConfig):
    data_dir = os.path.join(get_data_dir(), configfile.dataset_name)
    for urlinfo in configfile.urlinfos:
        _fetch_file(urlinfo.url, data_dir, filenames=urlinfo.filenames,
                    uncompress=urlinfo.opts['uncompress'])


def _fetch_file(url, data_dir, filenames=None, overwrite=False,
                md5sum=None, uncompress=True):
    """
    fetching function that load the requested file,
    downloading it if needed or requested


    IF the downloaded file is compressed, then the fetcher
    looks also for the uncompressed files before downloading .
    This only works for a one-to-one/one-to many configuration when one zipped
     file gives one or several uncompressed files. but we don't handle
     many-to-many yet

    maybe a better behavior is possible
    :param url:
    :param data_dir:

    :return:

    NOTES: NON-implemented nilearn parameters:
    * the resume option, that would resume partially downloaded files
    * username/password
    *handlers
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


def get_data_dir(dataset_name=None):
    """ Returns the directories in which nilearn looks for data.
    This is typically useful for the end-user to check where the data is
    downloaded and stored.

    *** taken from nilearn.datasets.utils ***
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    if dataset_name is not None:
        data_dir = os.path.join(data_dir, dataset_name)
    return data_dir


if __name__ == '__main__':
    pass
    # fetch_dataset(medical_charge_config)
