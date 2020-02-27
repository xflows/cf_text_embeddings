import logging
from enum import Enum
from os import path

import tensorflow as tf
import tensorflow_hub as hub
import tf_sentencepiece  # NOQA # pylint: disable=unused-import
import torch
import transformers
from elmoformanylangs import Embedder
from gensim import corpora
from gensim.corpora import Dictionary
from gensim.models import LsiModel, TfidfModel
from gensim.models.doc2vec import Doc2Vec
from gensim.models.keyedvectors import (FastTextKeyedVectors,
                                        Word2VecKeyedVectors)

import numpy as np
from cf_text_embeddings.base.common import (PROJECT_DATA_DIR,
                                            cf_text_embeddings_package_path)

# disable logs because they are output as messages in clowdflows
logging.getLogger('gensim').setLevel(logging.ERROR)
logging.getLogger('elmoformanylangs').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
tf.get_logger().setLevel('ERROR')


class AggregationMethod(Enum):
    average = 'average'
    summation = 'summation'


def cf_text_embeddings_model_path(lang, model_name):
    models_dir = cf_text_embeddings_package_path() if lang == 'test' else PROJECT_DATA_DIR
    lang = '' if lang is None else lang
    model_name = '' if model_name is None else model_name
    return path.join(models_dir, 'models', lang, model_name)


class EmbeddingsModelBase:
    def __init__(self, lang):
        self._lang = lang
        self._model_name = self.extract_model_name(lang)
        if self._model_name is None:
            raise Exception('Model for %s language is not supported' % lang)
        self._path = cf_text_embeddings_model_path(self._lang, self._model_name)
        self._model = None
        self._tfidf_dict = None
        self._tfidf_model = None

    def _load_model(self):
        return Word2VecKeyedVectors.load(self._path, mmap='r')

    @staticmethod
    def supported_models():
        return {}

    @classmethod
    def extract_model_name(cls, lang):
        supported_models = cls.supported_models()
        return supported_models.get(lang)

    @staticmethod
    def _tokens_to_embeddings(model, documents_tokens):
        embeddings = []
        for document_tokens in documents_tokens:
            n_document_tokens = len(document_tokens) or 1
            document_embeddings = np.zeros((n_document_tokens, model.vector_size))
            for i, token in enumerate(document_tokens):
                if token not in model.vocab:
                    continue
                document_embeddings[i] = model[token]
            embeddings.append(document_embeddings)
        return embeddings

    def _train_tfidf(self, documents_tokens):
        self._tfidf_dict = Dictionary(documents_tokens)  # fit dictionary
        corpus = [
            self._tfidf_dict.doc2bow(document_tokens) for document_tokens in documents_tokens
        ]  # convert corpus to BoW format
        self._tfidf_model = TfidfModel(corpus)  # fit model

    def _multiply_embeddings_with_tfidf(self, documents_tokens, embeddings):
        for i, document_tokens in enumerate(documents_tokens):
            if not document_tokens:
                continue
            # convert document_tokens to gensim corpus format (idx, freq=1)
            document_corpus = [(idx, 1) for idx in self._tfidf_dict.doc2idx(document_tokens)]
            # extract tfidf weights
            tfidf_vector = np.array([[el[1] for el in self._tfidf_model[document_corpus]]])
            tfidf_vector = tfidf_vector.reshape(-1, 1)
            embeddings[i] *= tfidf_vector
        return embeddings

    @staticmethod
    def _aggregate_embeddings(embeddings, aggregation_method):
        if aggregation_method == AggregationMethod.average.value:
            return np.array([np.average(embedding, axis=0) for embedding in embeddings])
        if aggregation_method == AggregationMethod.summation.value:
            return np.array([np.sum(embedding, axis=0) for embedding in embeddings])
        raise Exception('%s aggregation method is not supported' % aggregation_method)

    def apply(self, texts, aggregation_method=AggregationMethod.average.value,
              weighting_method=None):
        try:
            self._model = self._load_model()
        except FileNotFoundError:
            raise FileNotFoundError('Cannot find the model. Download it, if available.')
        embeddings = self._tokens_to_embeddings(self._model, texts)
        if weighting_method == 'tfidf':
            self._train_tfidf(texts)
            embeddings = self._multiply_embeddings_with_tfidf(texts, embeddings)
        embeddings = self._aggregate_embeddings(embeddings, aggregation_method)
        return embeddings

    def __getstate__(self):
        state = self.__dict__
        state.pop('_model')
        return state

    def __setstate__(self, state):
        # pylint: disable=W0201
        state['_model'] = None
        self.__dict__ = state


class EmbeddingsModelWord2Vec(EmbeddingsModelBase):
    @staticmethod
    def supported_models():
        return {
            # https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM
            'en': 'GoogleNews-vectors-negative300.wv.bin',
            # https://github.com/uchile-nlp/spanish-word-embeddings
            'es': 'SBW-vectors-300-min5.wv.bin',
            # http://vectors.nlpl.eu/repository/
            'sl': 'word2vec_si.wv',
            # http://vectors.nlpl.eu/repository/
            'hr': 'word2vec_hr.wv',
            # http://vectors.nlpl.eu/repository/
            'de': 'word2vec_de.wv',
            # http://vectors.nlpl.eu/repository/
            'ru': 'word2vec_ru.wv',
            # http://vectors.nlpl.eu/repository/
            'lv': 'word2vec_lv.wv',
            # http://vectors.nlpl.eu/repository/
            'ee': 'word2vec_ee.wv',
            'test': 'word2vec_test_model.bin'
        }


class EmbeddingsModelGloVe(EmbeddingsModelBase):
    @staticmethod
    def supported_models():
        return {
            # https://nlp.stanford.edu/projects/glove/
            'en': 'glove.6B.300d.wv.bin',
            # https://github.com/dccuchile/spanish-word-embeddings
            'es': 'glove_es.wv',
            # https://deepset.ai/german-word-embeddings
            'de': 'glove_de.wv',
            # internally glove models are handeled the same as word2vec
            'test': 'word2vec_test_model.bin'
        }


class EmbeddingsModelFastText(EmbeddingsModelBase):
    @staticmethod
    def supported_models():
        return {
            # https://github.com/facebookresearch/MUSE
            'en': 'wiki.multi.en.wv',
            # https://github.com/facebookresearch/MUSE
            'es': 'wiki.multi.es.wv',
            # https://github.com/facebookresearch/MUSE
            'de': 'wiki.multi.de.wv',
            # https://github.com/facebookresearch/MUSE
            'ru': 'wiki.multi.ru.wv',
            # https://fasttext.cc/docs/en/crawl-vectors.html
            'lv': 'fasttext_lv.wv',
            # https://fasttext.cc/docs/en/crawl-vectors.html
            'lt': 'fasttext_lt.wv',
            # https://github.com/facebookresearch/MUSE
            'ee': 'wiki.multi.ee.wv',
            # https://github.com/facebookresearch/MUSE
            'sl': 'wiki.multi.sl.wv',
            # https://github.com/facebookresearch/MUSE
            'hr': 'wiki.multi.hr.wv',
        }

    def _load_model(self):
        return FastTextKeyedVectors.load(self._path, mmap='r')


class EmbeddingsModelDoc2Vec(EmbeddingsModelBase):
    @staticmethod
    def supported_models():
        return {
            # https://github.com/jhlau/doc2vec
            'en': 'doc2vec.bin',
        }

    def _load_model(self):
        return Doc2Vec.load(self._path, mmap='r')

    def _tokens_to_embeddings(self, model, documents_tokens):
        document_embeddings = np.zeros((len(documents_tokens), model.vector_size))
        for i, document_tokens in enumerate(documents_tokens):
            if not document_tokens:
                continue
            document_embeddings[i] = model.infer_vector(document_tokens)
        return document_embeddings

    @staticmethod
    def _aggregate_embeddings(embeddings, aggregation_method):
        return embeddings


class EmbeddingsModelTensorFlow(EmbeddingsModelBase):
    def _load_model(self):
        path_ = cf_text_embeddings_model_path(self._lang, self._model_name)
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
    def _extract_tensors(model, document_tokens):
        tf_embeddings = model(document_tokens)
        return tf_embeddings

    def _tokens_to_embeddings(self, model, documents_tokens):
        embeddings = []
        for document_tokens in documents_tokens:
            if not document_tokens:
                document_tokens.append('')
            tf_embeddings = self._extract_tensors(model, document_tokens)
            document_embedding = self._extract_embeddings(tf_embeddings)
            embeddings.append(document_embedding)
        return embeddings


class EmbeddingsModelElmo(EmbeddingsModelBase):
    @staticmethod
    def supported_models():
        return {
            # http://vectors.nlpl.eu/repository/#
            'en': 'elmo',
            # http://vectors.nlpl.eu/repository/#
            'sl': 'elmo',
            # http://vectors.nlpl.eu/repository/#
            'es': 'elmo',
            # http://vectors.nlpl.eu/repository/#
            'ru': 'elmo',
            # http://vectors.nlpl.eu/repository/#
            'hr': 'elmo',
            # http://vectors.nlpl.eu/repository/#
            'ee': 'elmo',
            # http://vectors.nlpl.eu/repository/#
            'lv': 'elmo',
            # http://vectors.nlpl.eu/repository/#
            'de': 'elmo',
        }

    def _load_model(self):
        return Embedder(self._path)

    def _tokens_to_embeddings(self, model, documents_tokens):
        embeddings = np.array(model.sents2elmo(documents_tokens))
        return embeddings


class EmbeddingsModelHuggingface(EmbeddingsModelBase):
    def __init__(self, model_class, tokenizer_class, pretrained_weights, vector_size, max_seq):
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.pretrained_weights = pretrained_weights
        self._max_seq = max_seq
        self._vector_size = vector_size
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        self._tokenizer = self.tokenizer_class.from_pretrained(self.pretrained_weights)
        self._model = self.model_class.from_pretrained(self.pretrained_weights)
        return self._model

    @staticmethod
    def tokenize_text(tokenizer, texts, max_seq):
        return [tokenizer.encode(text, add_special_tokens=True)[:max_seq] for text in texts]

    @staticmethod
    def pad_text(tokenized_text, max_seq):
        return np.array([el + [0] * (max_seq - len(el)) for el in tokenized_text])

    @classmethod
    def tokenize_and_pad_text(cls, tokenizer, texts, max_seq):
        tokenized_text = cls.tokenize_text(tokenizer, texts, max_seq)
        padded_text = cls.pad_text(tokenized_text, max_seq)
        return torch.tensor(padded_text)

    def _tokens_to_embeddings(self, model, documents_tokens):
        default_embedding = np.zeros((1, self._vector_size))
        embeddings = []
        for document_tokens in documents_tokens:
            input_ids = self.tokenize_and_pad_text(self._tokenizer, [document_tokens],
                                                   self._max_seq)
            results = model(input_ids)[0]
            document_embedding = results.cpu().detach().numpy().squeeze(axis=0)
            document_embedding = (document_embedding
                                  if document_embedding.size > 0 else default_embedding)
            embeddings.append(document_embedding)
        return embeddings


class EmbeddingsModelBert(EmbeddingsModelHuggingface):
    def __init__(self, model_class=transformers.BertModel,
                 tokenizer_class=transformers.BertTokenizer, pretrained_weights='bert-base-uncased',
                 vector_size=768, max_seq=100):
        # Model source: https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1
        super().__init__(model_class, tokenizer_class, pretrained_weights, vector_size, max_seq)


class EmbeddingsModelLSI(EmbeddingsModelBase):
    def __init__(self, num_topics, decay):
        self.num_topics = num_topics
        self.decay = decay

    def apply(self, text, aggregation_method, weighting_method):
        dct = corpora.Dictionary(text)
        corpus = [dct.doc2bow(line) for line in text]
        self._model = LsiModel(corpus=corpus, id2word=dct, num_topics=self.num_topics,
                               decay=self.decay)
        embeddings = np.array([[el[1] for el in self._model[document]] for document in corpus])
        return embeddings
