# cf_text_embeddings

This is a [ClowdFlows 2.0](https://github.com/xflows/clowdflows-backend) package which contains widgets for word, sentence and document embeddings.

## Installation
Download this repository and run `python setup.py install` to install the package.
To enable the package in the ClowdFlows Backend, edit the `local_settings.py` file like shown below:
```
PACKAGE_TREE = [
    {
        "name": "Text Embeddings",
        "packages": ['text_embeddings'],
        "order": 1
    }
]
```
Then run ` ./manage.py import_package text_embeddings` from the ClowdFlows Backend directory to import widgets into the platform.
