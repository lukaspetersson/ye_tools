from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
        'torch == 1.11.0',
        'matplotlib == 3.5.2',
        ]

setup(
    name='trainer',
    version='0.1',
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    description='LSTM melody generator.'
)
