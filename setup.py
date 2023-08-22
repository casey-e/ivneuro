
# Import required functions
from setuptools import setup, find_packages


# Read the contents of README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Call setup function
setup(
    author="Eric Casey",
    author_email = 'e.toccalino@gmail.com',
    description="Tools for processing and analyzing neurophysiological signals recorded in-vivo.",
    name="ivneuro",
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(include=["ivneuro", "ivneuro.*"]),
    license='MIT',
    version="0.1.3",
    install_requires = ['numpy >= 1.24.1','pandas >= 2.0.1', 'matplotlib >= 3.6.2','scipy >= 1.10'],
    python_requires = '>=3.11.3',
    project_urls = {'Homepage':'https://github.com/casey-e/ivneuro'},
    url = 'https://github.com/casey-e/ivneuro',
)