
# Import required functions
from setuptools import setup, find_packages

# Call setup function
setup(
    author="Eric Casey",
    description="Tools for processing and analyzing neurophysiological signals recorded in-vivo.",
    name="ivneuro",
    packages=find_packages(include=["ivneuro", "ivneuro.*"]),
    version="0.1.0",
    install_requires = ['numpy >= 1.24.1','pandas >= 2.0.1', 'matplotlib >= 3.6.2','scipy >= 1.10'],
    python_requires = '>=3.11.3',
)