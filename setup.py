from setuptools import find_packages, setup

setup(
    name='pandavision',
    package_dir={'': 'app'},
    packages=find_packages(where='app'),
)

