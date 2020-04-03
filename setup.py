import os
import re

from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()
    # remove multiple spaces because of pypi requirement
    README = re.sub(' +', '', README)

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='cf_text_embeddings',
    version='0.1.12',
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


def download_nltk_requirements():
    import nltk  # NLTK should be imported after it is installed
    nltk.download('punkt')


def post_install():
    download_nltk_requirements()


post_install()
