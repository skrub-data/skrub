"""Custom build step: copy doc/_build/markdown → skrub/_docs before packaging."""

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithDocs(build_py):
    """Extend build_py to bundle the pre-built markdown documentation."""

    def run(self):
        self._copy_markdown_docs()
        super().run()

    def _copy_markdown_docs(self):
        source = Path("doc/_build/markdown")
        dest = Path("skrub/_docs")
        if not source.is_dir():
            return
        if dest.is_dir():
            shutil.rmtree(dest)
        shutil.copytree(source, dest)


setup(cmdclass={"build_py": BuildPyWithDocs})
