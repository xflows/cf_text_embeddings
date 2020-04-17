import os
import re

from setuptools import setup

with open('cf_text_embeddings/version.py', "rt") as f:
    version_content = f.read()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    mo = re.search(VSRE, version_content, re.M)
    if mo:
        version_string = mo.group(1)
    else:
        raise RuntimeError("Unable to find version string")

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
    version=version_string,
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
    try:
        import nltk  # NLTK should be imported after it is installed
        nltk.download('punkt')
    except Exception:
        print(('Warning: NLTK punkt languages weren\'t downloaded,'
               'run: python -m nltk.downloader punkt'))


def post_install():
    download_nltk_requirements()


post_install()
