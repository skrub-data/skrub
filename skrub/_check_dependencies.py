from importlib.metadata import PackageNotFoundError, requires, version

from packaging.requirements import Requirement


def check_dependencies():
    package_name = "skrub"
    package_version = version(package_name)
    requirements = requires(package_name)

    for req in requirements:
        req = Requirement(req)

        if req.marker is not None:
            # skip extra requirements
            continue

        try:
            installed_dep = version(req.name)
            if not req.specifier.contains(installed_dep, prereleases=True):
                raise ImportError(
                    f"{package_name} {package_version} requires {req!s} but you have"
                    f" {req.name} {installed_dep} installed, which is incompatible."
                )

        except PackageNotFoundError:
            raise ImportError(
                f"{package_name} {package_version} requires {req!s}, "
                "which you don't have installed."
            )
