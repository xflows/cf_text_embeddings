from enum import Enum
from os import path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from gensim.models.doc2vec import Doc2Vec
from gensim.models.keyedvectors import (FastTextKeyedVectors,
                                        Word2VecKeyedVectors)
from Orange.data import ContinuousVariable, Domain, Table


class ModelType(Enum):
    word2vec = 'word2vec'
    glove = 'glove'
    doc2vec = 'doc2vec'
    fasttext = 'fasttext'
    universal_sentence_encoder = 'universal_sentence_encoder'
    elmo = 'elmo'


GENSIM_MODELS = {ModelType.word2vec, ModelType.glove, ModelType.doc2vec, ModelType.fasttext}
TENSORFLOW_MODELS = {ModelType.universal_sentence_encoder, ModelType.elmo}


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
    if model_type in {ModelType.universal_sentence_encoder, ModelType.elmo}:
        return hub.Module(path_)
    raise Exception('%s model not supported' % model_type)


def text_embeddings_tokens_to_embeddings_gensim(model, tokens):
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


def text_embeddings_tokens_to_embeddings_tensorflow(model, tokens, **kwargs):
    model_output = None
    if 'model_output' in kwargs:
        model_output = kwargs.pop('model_output')

    tf_embeddings = model(tokens, **kwargs)
    if model_output is not None:
        tf_embeddings = tf_embeddings[model_output]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        embeddings = sess.run(tf_embeddings)

    if len(embeddings.shape) > 2:
        # convert elmo's arrays shape from (n, 1, m) to (n, m)
        embeddings = embeddings.squeeze()
    return embeddings


def text_embeddings_assign_embeddings_function(model_type):
    if model_type in GENSIM_MODELS:
        return text_embeddings_tokens_to_embeddings_gensim
    if model_type in TENSORFLOW_MODELS:
        return text_embeddings_tokens_to_embeddings_tensorflow
    raise Exception('%s model is not supported' % model_type.value)


def text_embeddings_words_to_orange_domain(n_features):
    return Domain([ContinuousVariable.make('Feature %d' % i) for i in range(n_features)])


def text_embeddings_extract_tokens(documents, token_annotation):
    tokens = []
    for document in documents:
        annotations_with_text = document.get_annotations_with_text(token_annotation)
        tokens.extend([ann[1] for ann in annotations_with_text])
    return tokens


def text_embeddings_apply_model(embeddings_function, model, documents, token_annotation, **kwargs):
    tokens = text_embeddings_extract_tokens(documents, token_annotation=token_annotation)
    embeddings = embeddings_function(model, tokens, **kwargs)
    domain = text_embeddings_words_to_orange_domain(embeddings.shape[1])
    return Table(domain, embeddings)


def text_embeddings_word2vec(input_dict):
    lang = input_dict['lang']
    adc = input_dict['adc']
    token_annotation = input_dict['token_annotation'] or 'Token'
    documents = adc.documents
    model_type = ModelType.word2vec
    embeddings_function = text_embeddings_assign_embeddings_function(model_type)

    model = None
    if lang == 'en':
        model = text_embeddings_load_model(model_type, 'GoogleNews-vectors-negative300.wv.bin')
    elif lang == 'es':
        model = text_embeddings_load_model(model_type, 'SBW-vectors-300-min5.wv.bin')

    if model is None:
        raise Exception('%s model for %s language is not supported' % (model_type, lang))

    bow_dataset = text_embeddings_apply_model(embeddings_function, model, documents,
                                              token_annotation=token_annotation)
    return {'bow_dataset': bow_dataset}


def text_embeddings_glove(input_dict):
    lang = input_dict['lang']
    adc = input_dict['adc']
    token_annotation = input_dict['token_annotation'] or 'Token'
    documents = adc.documents
    model_type = ModelType.glove
    embeddings_function = text_embeddings_assign_embeddings_function(model_type)

    model = None
    if lang == 'en':
        model = text_embeddings_load_model(model_type, 'glove.6B.300d.wv.bin')
    elif lang == 'es':
        model = None

    if model is None:
        raise Exception('%s model for %s language is not supported' % (model_type, lang))

    bow_dataset = text_embeddings_apply_model(embeddings_function, model, documents,
                                              token_annotation=token_annotation)
    return {'bow_dataset': bow_dataset}


def text_embeddings_fasttext(input_dict):
    lang = input_dict['lang']
    adc = input_dict['adc']
    token_annotation = input_dict['token_annotation'] or 'Token'
    documents = adc.documents
    model_type = ModelType.fasttext
    embeddings_function = text_embeddings_assign_embeddings_function(model_type)

    model = None
    if lang == 'en':
        # TODO include real english fasttext model
        model = text_embeddings_load_model(model_type, 'fasttext-small.bin')
    elif lang == 'es':
        model = None

    if model is None:
        raise Exception('%s model for %s language is not supported' % (model_type, lang))

    bow_dataset = text_embeddings_apply_model(embeddings_function, model, documents,
                                              token_annotation=token_annotation)
    return {'bow_dataset': bow_dataset}


def text_embeddings_universal_sentence_encoder(input_dict):
    lang = input_dict['lang']
    adc = input_dict['adc']
    token_annotation = input_dict['token_annotation'] or 'Sentence'
    documents = adc.documents
    model_type = ModelType.universal_sentence_encoder
    embeddings_function = text_embeddings_assign_embeddings_function(model_type)

    model = None
    if lang == 'en':
        model = text_embeddings_load_model(model_type, 'english')
    elif lang == 'es':
        model = None

    if model is None:
        raise Exception('%s model for %s language is not supported' % (model_type, lang))

    bow_dataset = text_embeddings_apply_model(embeddings_function, model, documents,
                                              token_annotation=token_annotation)
    return {'bow_dataset': bow_dataset}


def text_embeddings_doc2vec(input_dict):
    lang = input_dict['lang']
    adc = input_dict['adc']
    token_annotation = input_dict['token_annotation'] or 'TextBlock'
    documents = adc.documents
    model_type = ModelType.doc2vec
    embeddings_function = text_embeddings_assign_embeddings_function(model_type)

    model = None
    if lang == 'en':
        model = text_embeddings_load_model(model_type, 'doc2vec.bin')
    elif lang == 'es':
        model = None

    if model is None:
        raise Exception('%s model for %s language is not supported' % (model_type, lang))

    bow_dataset = text_embeddings_apply_model(embeddings_function, model, documents,
                                              token_annotation=token_annotation)
    return {'bow_dataset': bow_dataset}


def text_embeddings_elmo(input_dict):
    lang = input_dict['lang']
    adc = input_dict['adc']
    token_annotation = input_dict['token_annotation'] or 'TextBlock'
    documents = adc.documents
    model_type = ModelType.elmo
    embeddings_function = text_embeddings_assign_embeddings_function(model_type)

    model = None
    if lang == 'en':
        model = text_embeddings_load_model(model_type, 'english')
    elif lang == 'es':
        model = None

    if model is None:
        raise Exception('%s model for %s language is not supported' % (model_type, lang))

    bow_dataset = text_embeddings_apply_model(
        embeddings_function, model, documents, token_annotation=token_annotation,
        model_output='elmo', signature="default", as_dict=True)
    return {'bow_dataset': bow_dataset}
