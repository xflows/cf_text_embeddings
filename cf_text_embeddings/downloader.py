import argparse
import glob
import zipfile
from os import path, remove
from pathlib import Path
import gitlab
import requests

from bert_embedding import BertEmbedding

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


def download_bert_model():
    BertEmbedding(model='bert_12_768_12', dataset_name='wiki_multilingual_cased')


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


def download_models_for_language(project, module_dir, language):
    print('Downloading models for %s' % language)

    models_tree = project.repository_tree(path='models/%s' % language, ref=REF)
    n_models = len(models_tree)

    language_dir = path.join(module_dir, 'models/%s' % language)
    zipped_models = [f for f in glob.glob(path.join(language_dir, "*.zip"))]
    downloaded_models = [path.basename(model_path) for model_path in zipped_models]

    for i, model_tree in enumerate(models_tree, 1):
        model_name = model_tree['name']
        gitlab_path = model_tree['path']
        file_path = path.join(module_dir, gitlab_path)
        if model_name not in downloaded_models:
            print('%d/%d Downloading model %s to %s' % (i, n_models, gitlab_path, file_path))
            download_model(gitlab_path, file_path)
            zipped_models.append(file_path)
    return zipped_models


def process_models_for_language(project, module_dir, language):
    file_paths = download_models_for_language(project, module_dir, language)
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
    module_dir = path.join(str(Path.home()), '.cf_text_embeddings')

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
                process_models_for_language(project, module_dir, language)
        else:
            process_models_for_language(project, module_dir, language)
        download_bert_model()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
