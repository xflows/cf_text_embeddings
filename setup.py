import os

from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as readme:
    README = readme.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

setup(
    name='cf_text_embeddings',
    version='0.0.1',
    packages=['text_embeddings'],
    include_package_data=True,
    license='MIT License',
    description='Text Embeddings for ClowdFlows',
    long_description=README,
    url='https://github.com/anzev/rdm',
    author='Roman Orac',
    author_email='orac.roman@gmail.com',
    install_requires=requirements,
)
