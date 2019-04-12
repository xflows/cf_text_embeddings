from enum import Enum
from os import path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from gensim.models.doc2vec import Doc2Vec
from gensim.models.keyedvectors import (FastTextKeyedVectors,
                                        Word2VecKeyedVectors)
from Orange.data import ContinuousVariable, Domain, Table


def text_embeddings_package_folder_path():
    return path.dirname(path.realpath(__file__))


def text_embeddings_models_folder_path():
    return path.join(text_embeddings_package_folder_path(), 'models')


def text_embeddings_model_path(model_type, model_name):
    return path.join(text_embeddings_models_folder_path(), model_type, model_name)


def text_embeddings_words_to_orange_domain(n_features):
    return Domain([ContinuousVariable.make('Feature %d' % i) for i in range(n_features)])


class ModelType(Enum):
    word2vec = 'word2vec'
    glove = 'glove'
    doc2vec = 'doc2vec'
    fasttext = 'fasttext'
    universal_sentence_encoder = 'universal_sentence_encoder'
    elmo = 'elmo'


GENSIM_MODELS = {ModelType.word2vec, ModelType.glove, ModelType.doc2vec, ModelType.fasttext}
TENSORFLOW_MODELS = {ModelType.universal_sentence_encoder, ModelType.elmo}


class EmbeddingsModel:
    def __init__(self, model_type, model_name, **kwargs):
        self.model_type = model_type
        self.model_name = model_name
        self.kwargs = kwargs
        self._model = None

    def _load_model(self):
        path_ = text_embeddings_model_path(self.model_type.value, self.model_name)
        if self.model_type == ModelType.doc2vec:
            return Doc2Vec.load(path_, mmap='r')
        if self.model_type == ModelType.fasttext:
            return FastTextKeyedVectors.load(path_, mmap='r')
        if self.model_type in {ModelType.word2vec, ModelType.glove}:
            return Word2VecKeyedVectors.load(path_, mmap='r')
        if self.model_type in {ModelType.universal_sentence_encoder, ModelType.elmo}:
            return hub.Module(path_)
        raise Exception('%s model not supported' % self.model_type)

    @staticmethod
    def _extract_tokens(documents, token_annotation):
        tokens = []
        for document in documents:
            annotations_with_text = document.get_annotations_with_text(token_annotation)
            tokens.extend([ann[1] for ann in annotations_with_text])
        return tokens

    @classmethod
    def _assign_embeddings_function(cls, model_type):
        if model_type in GENSIM_MODELS:
            return cls._tokens_to_embeddings_gensim
        if model_type in TENSORFLOW_MODELS:
            return cls._tokens_to_embeddings_tensorflow
        raise Exception('%s model is not supported' % model_type.value)

    @staticmethod
    def _tokens_to_embeddings_gensim(model, tokens):
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

    @staticmethod
    def _tokens_to_embeddings_tensorflow(model, tokens, **kwargs):
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

    def apply(self, documents, token_annotation):
        self._model = self._load_model()
        tokens = self._extract_tokens(documents, token_annotation=token_annotation)
        embeddings_function = self._assign_embeddings_function(self.model_type)
        embeddings = embeddings_function(self._model, tokens, **self.kwargs)
        domain = text_embeddings_words_to_orange_domain(embeddings.shape[1])
        return Table(domain, embeddings)
