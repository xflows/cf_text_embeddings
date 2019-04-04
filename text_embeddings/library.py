from os import path

import numpy as np
from gensim.models import KeyedVectors


def text_embeddings_package_folder_path():
    return path.dirname(path.realpath(__file__))


def text_embeddings_models_folder_path():
    return path.join(text_embeddings_package_folder_path(), 'models')


def text_embeddings_model_path(model_type, model_name):
    return path.join(text_embeddings_models_folder_path(), model_type, model_name)


def text_embeddings_load_word2vec_model(model_type, model_name):
    path_ = text_embeddings_model_path(model_type, model_name)
    return KeyedVectors.load(path_, mmap='r')


def text_embeddings_words_to_embeddings(model, string):
    unknown_word_embedding = np.zeros(model.vector_size)
    embeddings = []
    words = string.split()
    for word in words:
        if word not in model.wv.vocab:
            embeddings.append(unknown_word_embedding)
            continue
        embedding = model.wv[word]
        embeddings.append(embedding)
    return words, np.array(embeddings)


def text_embeddings_word2vec(input_dict):
    lang = input_dict['lang']
    string = input_dict['string'] or ''

    model = None
    if lang == 'en':
        model = text_embeddings_load_word2vec_model('word2vec',
                                                    'GoogleNews-vectors-negative300.wv.bin')
    elif lang == 'es':
        model = text_embeddings_load_word2vec_model('word2vec', 'SBW-vectors-300-min5.wv.bin')

    if model is None:
        raise Exception('word2vec model for %s language is not supported' % lang)

    words, embeddings = text_embeddings_words_to_embeddings(model, string)
    return {'words': words, 'data': embeddings}
