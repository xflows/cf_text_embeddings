from os import path

import numpy as np
from gensim.models import KeyedVectors
from Orange.data import ContinuousVariable, Domain, Table


def text_embeddings_package_folder_path():
    return path.dirname(path.realpath(__file__))


def text_embeddings_models_folder_path():
    return path.join(text_embeddings_package_folder_path(), 'models')


def text_embeddings_model_path(model_type, model_name):
    return path.join(text_embeddings_models_folder_path(), model_type, model_name)


def text_embeddings_load_gensim_model(model_type, model_name):
    path_ = text_embeddings_model_path(model_type, model_name)
    return KeyedVectors.load(path_, mmap='r')


def text_embeddings_tokens_to_embeddings(model, tokens):
    unknown_word_embedding = np.zeros(model.vector_size)
    embeddings = []
    for token in tokens:
        if token not in model.wv.vocab:
            embeddings.append(unknown_word_embedding)
            continue
        embedding = model.wv[token]
        embeddings.append(embedding)
    return np.array(embeddings)


def text_embeddings_words_to_orange_domain(n_features):
    return Domain([ContinuousVariable.make('Feature %d' % i) for i in range(n_features)])


def text_embeddings_extract_tokens(documents, selector='Token'):
    tokens = []
    for document in documents:
        annotations_with_text = document.get_annotations_with_text(selector)
        tokens.extend([ann[1] for ann in annotations_with_text])
    return tokens


def text_embeddings_apply_gensim_model(model, documents):
    tokens = text_embeddings_extract_tokens(documents)
    embeddings = text_embeddings_tokens_to_embeddings(model, tokens)
    domain = text_embeddings_words_to_orange_domain(embeddings.shape[1])
    return Table(domain, embeddings)


def text_embeddings_word2vec(input_dict):
    lang = input_dict['lang']
    adc = input_dict['adc']
    documents = adc.documents

    model = None
    if lang == 'en':
        model = text_embeddings_load_gensim_model('word2vec',
                                                  'GoogleNews-vectors-negative300.wv.bin')
    elif lang == 'es':
        model = text_embeddings_load_gensim_model('word2vec', 'SBW-vectors-300-min5.wv.bin')

    if model is None:
        raise Exception('word2vec model for %s language is not supported' % lang)

    bow_dataset = text_embeddings_apply_gensim_model(model, documents)
    return {'bow_dataset': bow_dataset}


def text_embeddings_glove(input_dict):
    lang = input_dict['lang']
    adc = input_dict['adc']
    documents = adc.documents

    model = None
    if lang == 'en':
        model = text_embeddings_load_gensim_model('GloVe', 'glove.6B.300d.wv.bin')
    elif lang == 'es':
        model = None

    if model is None:
        raise Exception('GloVe model for %s language is not supported' % lang)

    bow_dataset = text_embeddings_apply_gensim_model(model, documents)
    return {'bow_dataset': bow_dataset}
