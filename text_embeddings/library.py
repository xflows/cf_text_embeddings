from enum import Enum
from os import path

import numpy as np
from gensim.models.doc2vec import Doc2Vec
from gensim.models.keyedvectors import (FastTextKeyedVectors,
                                        Word2VecKeyedVectors)
from Orange.data import ContinuousVariable, Domain, Table

class EmbeddingsLibrary(Enum):
    gensim = 'gensim'
    tensorflow = 'tensorflow'


class ModelType(Enum):
    word2vec = 'word2vec'
    glove = 'glove'
    doc2vec = 'doc2vec'
    fasttext = 'fasttext'


def text_embeddings_package_folder_path():
    return path.dirname(path.realpath(__file__))


def text_embeddings_models_folder_path():
    return path.join(text_embeddings_package_folder_path(), 'models')


def text_embeddings_model_path(model_type, model_name):
    return path.join(text_embeddings_models_folder_path(), model_type, model_name)


def text_embeddings_load_model(model_type, model_name):
    path_ = text_embeddings_model_path(model_type.value, model_name)
    if model_type == ModelType.doc2vec:
        return Doc2Vec.load(path_, mmap='r')
    if model_type == ModelType.fasttext:
        return FastTextKeyedVectors.load(path_, mmap='r')
    if model_type in {ModelType.word2vec, ModelType.glove}:
        return Word2VecKeyedVectors.load(path_, mmap='r')
    raise Exception('%s model not supported' % model_type)


def text_embeddings_tokens_to_embeddings(model, tokens):
    unknown_word_embedding = np.zeros(model.vector_size)
    embeddings = []
    for token in tokens:
        if model.__class__ == Doc2Vec:
            sub_tokens = token.split(' ')  # TextBlock to list of strings
            embedding = model.infer_vector(sub_tokens)
        elif model.__class__ in {Word2VecKeyedVectors, FastTextKeyedVectors}:
            if token not in model.wv.vocab:
                embeddings.append(unknown_word_embedding)
                continue
            embedding = model.wv[token]
        else:
            raise Exception('%s model not supported' % model.__class__)
        embeddings.append(embedding)
    return np.array(embeddings)


def text_embeddings_words_to_orange_domain(n_features):
    return Domain([ContinuousVariable.make('Feature %d' % i) for i in range(n_features)])


def text_embeddings_extract_tokens(documents, selector):
    tokens = []
    for document in documents:
        annotations_with_text = document.get_annotations_with_text(selector)
        tokens.extend([ann[1] for ann in annotations_with_text])
    return tokens


def text_embeddings_apply_model(embeddings_library, model, documents, selector):
    tokens = text_embeddings_extract_tokens(documents, selector=selector)
    embeddings = text_embeddings_tokens_to_embeddings(model, tokens)
    domain = text_embeddings_words_to_orange_domain(embeddings.shape[1])
    return Table(domain, embeddings)


def text_embeddings_word2vec(input_dict):
    lang = input_dict['lang']
    adc = input_dict['adc']
    documents = adc.documents

    model = None
    if lang == 'en':
        model = text_embeddings_load_model(ModelType.word2vec,
                                           'GoogleNews-vectors-negative300.wv.bin')
    elif lang == 'es':
        model = text_embeddings_load_model(ModelType.word2vec, 'SBW-vectors-300-min5.wv.bin')

    if model is None:
        raise Exception('%s model for %s language is not supported' % (ModelType.word2vec, lang))

    bow_dataset = text_embeddings_apply_model(EmbeddingsLibrary.gensim, model, documents,
                                              selector='Token')
    return {'bow_dataset': bow_dataset}


def text_embeddings_glove(input_dict):
    lang = input_dict['lang']
    adc = input_dict['adc']
    documents = adc.documents

    model = None
    if lang == 'en':
        model = text_embeddings_load_model(ModelType.glove, 'glove.6B.300d.wv.bin')
    elif lang == 'es':
        model = None

    if model is None:
        raise Exception('%s model for %s language is not supported' % (ModelType.glove, lang))

    bow_dataset = text_embeddings_apply_model(EmbeddingsLibrary.gensim, model, documents,
                                              selector='Token')
    return {'bow_dataset': bow_dataset}


def text_embeddings_fasttext(input_dict):
    lang = input_dict['lang']
    adc = input_dict['adc']
    documents = adc.documents

    model = None
    if lang == 'en':
        # TODO include real english fasttext model
        model = text_embeddings_load_model(ModelType.fasttext, 'fasttext-small.bin')
    elif lang == 'es':
        model = None

    if model is None:
        raise Exception('%s model for %s language is not supported' % (ModelType.fasttext, lang))

    bow_dataset = text_embeddings_apply_model(EmbeddingsLibrary.gensim, model, documents,
                                              selector='Token')
    return {'bow_dataset': bow_dataset}


def text_embeddings_doc2vec(input_dict):
    lang = input_dict['lang']
    adc = input_dict['adc']
    documents = adc.documents

    model = None
    if lang == 'en':
        model = text_embeddings_load_model(ModelType.doc2vec, 'doc2vec.bin')
    elif lang == 'es':
        model = None

    if model is None:
        raise Exception('%s model for %s language is not supported' % (ModelType.doc2vec, lang))

    bow_dataset = text_embeddings_apply_model(EmbeddingsLibrary.gensim, model, documents,
                                              selector='TextBlock')
    return {'bow_dataset': bow_dataset}
