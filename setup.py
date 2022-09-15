from configparser import ConfigParser
import os
from setuptools import setup


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
    project_dir = config["metadata"]["name"]
    req_path = os.path.join(project_dir, ".requirements.txt")
    with open(req_path, "w+") as f:
        f.write(deps)


if __name__ == "__main__":
    setup_package()
