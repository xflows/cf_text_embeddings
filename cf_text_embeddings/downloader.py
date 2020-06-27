import argparse
import glob
import zipfile
from os import path, remove, rename

import gitlab
import transformers

import requests

from .base.common import PROJECT_DATA_DIR, ensure_dir

GITLAB_URL = 'https://repo.ijs.si/'
GITLAB_PROJECT = 'vpodpecan/cf_text_embeddings_models'
REF = 'master'


def download_model(gitlab_path, local_filename):
    url = path.join(GITLAB_URL, GITLAB_PROJECT, 'raw', REF, gitlab_path)
    filepath, filename = path.split(local_filename)
    temp_filename = path.join(filepath, filename + '_temp')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(temp_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        rename(temp_filename, local_filename)  # handle partialy downloaded files


def download_bert_model():
    print('Downloading Bert Model')
    model_class = transformers.BertModel
    pretrained_weights_list = [
        'bert-base-uncased',
        'bert-base-multilingual-uncased',
    ]
    for pretrained_weights in pretrained_weights_list:
        print('Downloading', pretrained_weights)
        model_class.from_pretrained(pretrained_weights)

    model_class = transformers.DistilBertModel
    model_class.from_pretrained('distilbert-base-multilingual-cased')


def unzip_file(zip_filepath):
    directory_to_extract_to = path.dirname(zip_filepath)
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)


def unzip_files(file_paths):
    for file_path in file_paths:
        print('Unzipping %s' % (file_path))
        unzip_file(file_path)


def remove_files(file_paths):
    for file_path in file_paths:
        print('Removing zip %s' % file_path)
        remove(file_path)


def download_models_for_language(project, models_dir, language):
    print('Downloading models for %s' % language)

    models_tree = project.repository_tree(path='models/%s' % language, ref=REF)
    n_models = len(models_tree)

    language_dir = path.join(models_dir, 'models', language)
    zipped_models = [f for f in glob.glob(path.join(language_dir, "*.zip"))]
    downloaded_models = [path.basename(model_path) for model_path in zipped_models]

    for i, model_tree in enumerate(models_tree, 1):
        model_name = model_tree['name']
        gitlab_path = model_tree['path']
        file_path = path.join(models_dir, gitlab_path)
        ensure_dir(file_path)
        if model_name not in downloaded_models:
            print('%d/%d Downloading model %s to %s' % (i, n_models, gitlab_path, file_path))
            download_model(gitlab_path, file_path)
            zipped_models.append(file_path)
    return zipped_models


def process_models_for_language(project, models_dir, language):
    file_paths = download_models_for_language(project, models_dir, language)
    unzip_files(file_paths)
    remove_files(file_paths)


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
    models_dir = PROJECT_DATA_DIR
    ensure_dir(models_dir)

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
                process_models_for_language(project, models_dir, language)
        else:
            process_models_for_language(project, models_dir, language)
        if language == 'multi':
            download_bert_model()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
