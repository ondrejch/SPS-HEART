"""Build hook: bundle ``config/`` into the package for wheels.

The project metadata lives in ``pyproject.toml``.  This file only extends
``build_py``, so that a wheel carries the configuration under
``na49_mixing/_data/config``.  ``na49_mixing._paths.nested_root()`` finds
it there when the package runs outside a checkout.
"""
import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py

HERE = Path(__file__).resolve().parent


class BuildPyWithConfig(build_py):
    def run(self):
        super().run()
        source = HERE / "config"
        if source.is_dir():
            target = Path(self.build_lib) / "na49_mixing" / "_data" / "config"
            shutil.copytree(source, target, dirs_exist_ok=True)


setup(cmdclass={"build_py": BuildPyWithConfig})
