from configparser import ConfigParser
import os
from setuptools import setup
from pathlib import Path as _Path


def setup_package():
    write_requirements()
    setup()


def write_requirements():
    """Parse dependencies from setup.cfg and write them
    within the project scope.

    Notes
    -----
    It will allow to compare installed versions against
    required versions at run time, during the import of this package.
    """
    setup_file = "setup.cfg"
    config = ConfigParser()
    config.read(setup_file)
    deps = config["options"]["install_requires"].strip()
    project_name = config["metadata"]["name"]
    project_dir = _Path(__file__).parent / project_name
    req_path = project_dir / "_config_requirements.py"
    with open(req_path, "w") as f:
        body = "deps = " + str({"deps": deps})
        f.write(body)


if __name__ == "__main__":
    setup_package()
