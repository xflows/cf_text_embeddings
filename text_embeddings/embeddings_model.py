from os import path

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from gensim.models.doc2vec import Doc2Vec
from gensim.models.keyedvectors import (FastTextKeyedVectors,
                                        Word2VecKeyedVectors)
from Orange.data import ContinuousVariable, DiscreteVariable, Domain, Table


def text_embeddings_package_folder_path():
    return path.dirname(path.realpath(__file__))


def text_embeddings_models_folder_path():
    return path.join(text_embeddings_package_folder_path(), 'models')


def text_embeddings_model_path(model_name):
    return path.join(text_embeddings_models_folder_path(), model_name)


class EmbeddingsModelBase:
    def __init__(self, model_name, default_token_annotation=None):
        self._model_name = model_name
        self._path = text_embeddings_model_path(self._model_name)
        self._model = None
        self._token_annotation = None
        self.default_token_annotation = default_token_annotation

    def _load_model(self):
        return Word2VecKeyedVectors.load(self._path, mmap='r')

    @staticmethod
    def _extract_tokens(documents, token_annotation):
        documents_tokens = []
        for document in documents:
            annotations_with_text = document.get_annotations_with_text(token_annotation)
            document_tokens = [ann[1] for ann in annotations_with_text]
            documents_tokens.append(document_tokens)
        return documents_tokens

    @staticmethod
    def _extract_labels(documents, binary=False):
        document_labels = [document.get_first_label() for document in documents]
        uniq_labels = list(sorted(set(document_labels)))
        if binary:
            return [uniq_labels.index(r) for r in document_labels], uniq_labels
        return document_labels, uniq_labels

    @staticmethod
    def _tokens_to_embeddings(model, documents_tokens):
        embeddings = []
        for document_tokens in documents_tokens:
            n_document_tokens = len(document_tokens) or 1
            document_embedding = np.zeros((n_document_tokens, model.vector_size))
            for i, token in enumerate(document_tokens):
                if token not in model.vocab:
                    continue
                document_embedding[i] = model[token]
            embeddings.append(np.average(document_embedding, axis=0))
        return np.array(embeddings)

    @staticmethod
    def _orange_domain(n_features, unique_labels):
        return Domain([ContinuousVariable.make('Feature %d' % i) for i in range(n_features)],
                      DiscreteVariable('class', values=unique_labels))

    def apply(self, documents, token_annotation):
        self._token_annotation = token_annotation
        self._model = self._load_model()
        documents_tokens = self._extract_tokens(documents, token_annotation=token_annotation)
        Y, unique_labels = self._extract_labels(documents, binary=True)
        embeddings = self._tokens_to_embeddings(self._model, documents_tokens)
        domain = self._orange_domain(embeddings.shape[1], unique_labels)
        table = Table(domain, embeddings, Y=Y)
        return table

    def __getstate__(self):
        state = self.__dict__
        state.pop('_model')
        return state

    def __setstate__(self, state):
        # pylint: disable=W0201
        state['_model'] = None
        self.__dict__ = state


class EmbeddingsModelWord2Vec(EmbeddingsModelBase):
    pass


class EmbeddingsModelGloVe(EmbeddingsModelBase):
    pass


class EmbeddingsModelFastText(EmbeddingsModelBase):
    def _load_model(self):
        return FastTextKeyedVectors.load(self._path, mmap='r')


class EmbeddingsModelDoc2Vec(EmbeddingsModelBase):
    def _load_model(self):
        return Doc2Vec.load(self._path, mmap='r')

    def _tokens_to_embeddings(self, model, documents_tokens):
        document_embeddings = np.zeros((len(documents_tokens), model.vector_size))
        for i, document_tokens in enumerate(documents_tokens):
            if not document_tokens:
                continue
            if self._token_annotation == 'TextBlock':
                document_tokens = document_tokens[0].split(' ')  # TextBlock to list of strings
            document_embeddings[i] = model.infer_vector(document_tokens)
        return document_embeddings


class EmbeddingsModelTensorFlow(EmbeddingsModelBase):
    def _load_model(self):
        path_ = text_embeddings_model_path(self._model_name)
        return hub.Module(path_)

    @staticmethod
    def _extract_embeddings(tf_embeddings):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            embeddings = sess.run(tf_embeddings)
        return embeddings


class EmbeddingsModelUniversalSentenceEncoder(EmbeddingsModelTensorFlow):
    @staticmethod
    def _extract_tensors(model, documents_tokens):
        tf_embeddings = model(documents_tokens)
        return tf_embeddings

    def _tokens_to_embeddings(self, model, documents_tokens):
        embeddings = []
        for document_tokens in documents_tokens:
            tf_embeddings = self._extract_tensors(model, document_tokens)
            document_embedding = self._extract_embeddings(tf_embeddings)
            embeddings.append(np.average(document_embedding, axis=0))
        return np.array(embeddings)


class EmbeddingsModelElmo(EmbeddingsModelTensorFlow):
    def __init__(self, model_name, model_output, signature, as_dict):
        super().__init__(model_name)
        self.model_output = model_output
        self.signature = signature
        self.as_dict = as_dict

    @staticmethod
    def _calculate_sequence_len(documents_tokens):
        return [len(document_tokens) for document_tokens in documents_tokens]

    @staticmethod
    def _pad_tokens(documents_tokens, max_len):
        for document_tokens in documents_tokens:
            n_pad = max_len - len(document_tokens)
            if n_pad == 0:
                continue
            for _ in range(n_pad):
                document_tokens.append('')
        return documents_tokens

    def _tokens_to_embeddings(self, model, documents_tokens):
        tf_embeddings = self._extract_tensors(model, documents_tokens)
        embeddings = self._extract_embeddings(tf_embeddings)
        return embeddings

    def _extract_tensors(self, model, documents_tokens):
        sequence_len = self._calculate_sequence_len(documents_tokens)
        max_len = max(sequence_len) or 1
        documents_tokens = self._pad_tokens(documents_tokens, max_len)
        tf_embeddings = model(inputs={
            "tokens": documents_tokens,
            "sequence_len": sequence_len
        }, signature=self.signature, as_dict=self.as_dict)[self.model_output]
        return tf_embeddings
