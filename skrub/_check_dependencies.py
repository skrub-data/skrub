from importlib.metadata import requires, version, PackageNotFoundError
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
            if not req.specifier.contains(installed_dep):
                raise ImportError(
                    f"{package_name} {package_version} requires {str(req)} "
                    f"but you have {req.name}={installed_dep}, which is not compatible."
                )

        except PackageNotFoundError:
            raise ImportError(
                f"{package_name} {package_version} requires {str(req)}, "
                "which you don't have."
            )
