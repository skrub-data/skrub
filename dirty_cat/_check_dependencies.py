import pkg_resources


def check_dependencies():
    package_name = "dirty-cat"
    env = pkg_resources.Environment()
    package = env[package_name][0]
    dependencies = package.requires()
    for dep in dependencies:
        installed_dep = env[dep.name][0]
        if installed_dep not in dep:
            raise ImportError(
                f"{package_name} {package.version} requires {dep} "
                f"but you have {installed_dep}, which is not compatible."
            )
