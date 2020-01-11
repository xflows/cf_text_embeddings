import os
import re

from setuptools import setup

from pipenv.project import Project
from pipenv.utils import convert_deps_to_pip

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()
    # remove multiple spaces because of pypi requirement
    README = re.sub(' +', '', README)

pfile = Project(chdir=False).parsed_pipfile
requirements = convert_deps_to_pip(pfile['packages'], r=False)

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='cf_text_embeddings',
    version='0.1.9',
    packages=['cf_text_embeddings'],
    include_package_data=True,
    license='MIT License',
    description='Text Embeddings for ClowdFlows',
    long_description=README,
    url='https://github.com/xflows/cf_text_embeddings',
    author='Roman Orac',
    author_email='orac.roman@gmail.com',
    install_requires=requirements,
)
