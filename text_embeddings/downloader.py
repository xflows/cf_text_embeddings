import argparse
import zipfile
from os import path, remove

import gitlab
import requests

import text_embeddings

GITLAB_URL = 'https://repo.ijs.si/'
GITLAB_PROJECT = 'vpodpecan/cf_text_embeddings_models'
REF = 'master'


def download_model(gitlab_path, local_filename):
    url = path.join(GITLAB_URL, GITLAB_PROJECT, 'raw', REF, gitlab_path)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)


def unzip_file(zip_filepath):
    directory_to_extract_to = path.dirname(zip_filepath)
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


def remove_file(file_path):
    remove(file_path)


def download_models_for_language(project, module_dir, language):
    print('Downloading models for %s' % language)

    models_tree = project.repository_tree(path='models/%s' % language, ref=REF)
    n_models = len(models_tree)
    for i, model_tree in enumerate(models_tree, 1):
        gitlab_path = model_tree['path']
        file_path = path.join(module_dir, gitlab_path)
        print('%d/%d Downloading model %s to %s' % (i, n_models, gitlab_path, file_path))
        download_model(gitlab_path, file_path)

        print('Unzipping %s' % (file_path))
        unzip_file(file_path)

        print('Removing zip %s' % file_path)
        remove_file(file_path)
    print('Done')


def fetch_languages(project):
    languages_tree = project.repository_tree(path='models', ref=REF)
    languages = sorted([language_tree['name'] for language_tree in languages_tree])
    languages += ['all']
    return languages


def list_languages(languages):
    print('Available languages:')
    for i, language in enumerate(languages, 1):
        print('%d. %s' % (i, language))


def main():
    # anonymous gitlab instance, read-only for public resources
    gl = gitlab.Gitlab(GITLAB_URL)
    project = gl.projects.get(GITLAB_PROJECT)
    module_dir = path.dirname(text_embeddings.__file__)

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--list", action="store_true", help="list available languages")
    parser.add_argument("-d", "--download", type=str,
                        help="download models for certain language code")
    args = parser.parse_args()

    languages = fetch_languages(project)

    if args.list:
        list_languages(languages)
    elif args.download in languages:
        language = args.download
        if language == 'all':
            for language in languages:
                download_models_for_language(project, module_dir, language)
        else:
            download_models_for_language(project, module_dir, language)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
