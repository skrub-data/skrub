import pkg_resources


def check_dependencies():
    package_name = "skrub"
    env = pkg_resources.Environment()
    package = env[package_name][0]
    requirements = package.requires()
    for req in requirements:
        try:
            installed_dep = next(
                iter(
                    (
                        installed_dep
                        for installed_dep in env[req.name]
                        if installed_dep.project_name == req.name
                    )
                )
            )
        except StopIteration:
            raise ImportError(
                f"{package_name} {package.version} requires {req}, "
                "which you don't have."
            )

        if installed_dep not in req:
            raise ImportError(
                f"{package_name} {package.version} requires {req} "
                f"but you have {installed_dep}, which is not compatible."
            )
