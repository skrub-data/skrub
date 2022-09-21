"""
Check versions from .requirements.txt to raise error at init.
.requirements.txt is written from setup.cfg during setup.
"""
import re
from importlib.metadata import version
from pathlib import Path as _Path

from ._config_requirements import deps


def _check_pack_version(dep, package_name, required_version, sign):
    installed_version = version(package_name)
    error_msg = (
        f"dirty_cat {_version} requires {dep}, but you have "
        f"{package_name} {installed_version} which is incompatible."
    )
    try:
        from sklearn.externals._packaging.version import parse as parse_version
    except ModuleNotFoundError:
        raise ImportError(error_msg)
    parsed_installed_version = parse_version(installed_version)
    parsed_required_version = parse_version(required_version)
    if not eval(f"'{parsed_installed_version}' {sign} '{parsed_required_version}'"):
        raise ImportError(error_msg)


parent_dir = _Path(__file__).parent
with open(parent_dir / "VERSION.txt") as _fh:
    _version = _fh.read().strip()

deps = deps["deps"].split("\n")
for dep in deps:
    dep = dep.strip()
    matches_package = re.findall(r"^[\sa-zA-Z0-9-]+", dep)
    package_name = matches_package[0]
    signs = ["<", "<=", ">", ">=", "==", "!="]
    for sign in signs:
        pattern = rf"{sign}[a-zA-Z0-9.*]+"
        matches_version = re.findall(pattern, dep)
        if len(matches_version) > 0:
            required_version = matches_version[0].replace(sign, "")
            _check_pack_version(dep, package_name, required_version, sign)
