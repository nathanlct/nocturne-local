import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

setup(
    name='nocturne',
    version='0.0.1',
    author='Nathan Lichtle, Eugene Vinitsky, and Xiaomeng Yang',
    packages=find_packages(),
)
