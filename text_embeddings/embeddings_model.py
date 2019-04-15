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


def text_embeddings_model_path(model_name):
    return path.join(text_embeddings_models_folder_path(), model_name)


def text_embeddings_words_to_orange_domain(n_features):
    return Domain([ContinuousVariable.make('Feature %d' % i) for i in range(n_features)])


class EmbeddingsModelBase:
    def __init__(self, model_name):
        self.model_name = model_name
        self.path_ = text_embeddings_model_path(self.model_name)
        self._model = None

    def _load_model(self):
        return Word2VecKeyedVectors.load(self.path_, mmap='r')

    @staticmethod
    def _extract_tokens(documents, token_annotation):
        tokens = []
        for document in documents:
            annotations_with_text = document.get_annotations_with_text(token_annotation)
            tokens.extend([ann[1] for ann in annotations_with_text])
        return tokens

    @staticmethod
    def _tokens_to_embeddings(model, tokens):
        unknown_word_embedding = np.zeros(model.vector_size)
        embeddings = []
        for token in tokens:
            if token not in model.wv.vocab:
                embeddings.append(unknown_word_embedding)
                continue
            embedding = model.wv[token]
            embeddings.append(embedding)
        return np.array(embeddings)

    def apply(self, documents, token_annotation):
        self._model = self._load_model()
        tokens = self._extract_tokens(documents, token_annotation=token_annotation)
        embeddings = self._tokens_to_embeddings(self._model, tokens)
        domain = text_embeddings_words_to_orange_domain(embeddings.shape[1])
        return Table(domain, embeddings)


class EmbeddingsModelWord2Vec(EmbeddingsModelBase):
    pass


class EmbeddingsModelGloVe(EmbeddingsModelBase):
    pass


class EmbeddingsModelFastText(EmbeddingsModelBase):
    def _load_model(self):
        return FastTextKeyedVectors.load(self.path_, mmap='r')


class EmbeddingsModelDoc2Vec(EmbeddingsModelBase):
    def _load_model(self):
        return Doc2Vec.load(self.path_, mmap='r')

    def _tokens_to_embeddings(self, model, tokens):
        embeddings = []
        for token in tokens:
            sub_tokens = token.split(' ')  # TextBlock to list of strings
            embedding = model.infer_vector(sub_tokens)
            embeddings.append(embedding)
        return np.array(embeddings)


class EmbeddingsModelTensorFlow(EmbeddingsModelBase):
    def _load_model(self):
        path_ = text_embeddings_model_path(self.model_name)
        return hub.Module(path_)

    def _tokens_to_embeddings(self, model, tokens):
        tf_embeddings = model(tokens)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            embeddings = sess.run(tf_embeddings)
        return embeddings


class EmbeddingsModelUniversalSentenceEncoder(EmbeddingsModelTensorFlow):
    pass


class EmbeddingsModelElmo(EmbeddingsModelTensorFlow):
    def __init__(self, model_name, model_output, signature, as_dict):
        super().__init__(model_name)
        self.model_output = model_output
        self.signature = signature
        self.as_dict = as_dict

    def _tokens_to_embeddings(self, model, tokens):
        tf_embeddings = model(tokens, signature=self.signature,
                              as_dict=self.as_dict)[self.model_output]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            embeddings = sess.run(tf_embeddings)

        if len(embeddings.shape) > 2:
            # convert elmo's arrays shape from (n, 1, m) to (n, m)
            embeddings = embeddings.squeeze()
        return embeddings
