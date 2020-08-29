import unittest
import warnings
from os import path

import numpy as np

from ..base.common import PROJECT_DATA_DIR
from ..base.embeddings_model import (TFIDF, AggregationMethod,
                                     EmbeddingsModelBert, EmbeddingsModelElmo,
                                     EmbeddingsModelElmoAllen,
                                     EmbeddingsModelLSI,
                                     EmbeddingsModelUniversalSentenceEncoder,
                                     EmbeddingsModelWord2Vec)


def word_tokens():
    return [[
        'Nintendo', 'has', 'announced', 'that', 'two', 'of', 'its', 'Switch', 'games', 'will', 'be',
        'updated.'
    ], ['Although', 'there’s', 'a', 'fierce', 'rivalry', 'between', 'smartphone', 'manufacturers.']]


def word_tokens_repeated_words():
    return [['Nintendo', 'has', 'that', 'and', 'of', 'fierce', 'Switch', 'games', 'and'],
            ['Although', 'there’s', 'and', 'fierce', 'rivalry', 'between', 'Switch']]


def sentence_tokens():
    return [['This is sentence1 of doc1', 'this is sentence2 of doc1'],
            ['This is sentence1 of doc2', 'this is sentence2 of doc2'],
            ['This is sentence1 of doc3']]


def doc_tokens():
    return [
        'Nintendo has announced that two of its Switch games will be updated.',
        'Although there’s a fierce rivalry between smartphone manufacturers.'
    ]


class EmbeddingsModelTest(unittest.TestCase):
    def test_word2vec_model(self):
        documents_tokens = word_tokens()
        embeddings_model = EmbeddingsModelWord2Vec('test')
        actual_X = embeddings_model.apply(documents_tokens, AggregationMethod.average.value, None)

        expected_X = np.array([[0.039, 0.026], [-0.045, -0.072]])
        self.assertEqual(True, np.allclose(expected_X, actual_X, atol=1e-03))

    def test_word2vec_model_not_exists(self):
        self.assertRaises(Exception, EmbeddingsModelWord2Vec, 'not_exists')

    def test_lsi_model_bow(self):
        np.random.seed(42)
        text = word_tokens()
        embeddings_model = EmbeddingsModelLSI(2, 1, train_on_tfidf=False, filter_extremes=False)
        actual_X = embeddings_model.apply(text, AggregationMethod.average.value, None)
        expected_X = np.array([[3.464], [2.828]])
        self.assertEqual(True, np.allclose(expected_X, actual_X, atol=1e-03))

    def test_lsi_model_tfidf(self):
        np.random.seed(42)
        text = word_tokens()
        embeddings_model = EmbeddingsModelLSI(2, 1, train_on_tfidf=True, filter_extremes=False)
        actual_X = embeddings_model.apply(text, AggregationMethod.average.value, None)
        expected_X = np.array([[1.], [1.]])
        self.assertEqual(True, np.allclose(expected_X, actual_X, atol=1e-03))

    def test_bert_model(self):
        if not path.exists(PROJECT_DATA_DIR):
            # This test is not run by Travis CI
            warnings.warn(
                'Test test_bert_model not executed, because you need to download model first')
            return

        documents_tokens = doc_tokens()
        embeddings_model = EmbeddingsModelBert('bert-base-uncased')
        embeddings = embeddings_model.apply(documents_tokens,
                                            aggregation_method=AggregationMethod.average.value)
        actual_X = embeddings[:, :2]

        expected_X = np.array([[-0.22724754, -0.42256614], [0.28956912, -0.16998145]])
        self.assertEqual(True, np.allclose(expected_X, actual_X, atol=1e-03))

    def test_elmo_model(self):
        english_elmo = path.join(PROJECT_DATA_DIR, 'models', 'en', 'elmo')
        if not path.exists(english_elmo):
            warnings.warn(
                'Test test_elmo_model not executed, because you need to download english elmo model'
            )
            return
        sentences = sentence_tokens()
        model = EmbeddingsModelElmo('en')
        embeddings = model.apply(sentences, aggregation_method=AggregationMethod.average.value)

        # The model returns different outputs when changing batch size
        # This happens due to states in LSTM
        actual_X = embeddings[:, :2]
        expected_X = np.array([[-0.13834494, -0.26584834]])
        self.assertEqual(True, np.allclose(expected_X, actual_X[0, :], atol=1e-03))
        self.assertEqual(3, actual_X.shape[0])

    def test_elmo_allen_model(self):
        slovenian_elmo = path.join(PROJECT_DATA_DIR, 'models', 'sl', 'slovenian-elmo-embedia')
        if not path.exists(slovenian_elmo):
            warnings.warn(
                'Test test_elmo_allen_model not executed, you need to download slovenian elmo model'
            )
            return

        words = word_tokens()
        model = EmbeddingsModelElmoAllen('sl')
        embeddings = model.apply(words, aggregation_method=AggregationMethod.average.value)

        actual_X = embeddings[:, :2]
        expected_X = np.array([[-1.19756973, -0.62998658]])
        self.assertEqual(True, np.allclose(expected_X, actual_X[0, :], atol=1e-03))
        self.assertEqual(2, actual_X.shape[0])

    def test_use_model(self):
        use_en = path.join(PROJECT_DATA_DIR, 'models', 'en', 'universal_sentence_encoder_english')
        if not path.exists(use_en):
            warnings.warn(
                'Test test_use_model not executed, because you need to download english use model')
            return
        sentences = sentence_tokens()
        model = EmbeddingsModelUniversalSentenceEncoder('en')
        embeddings = model.apply(sentences, aggregation_method=AggregationMethod.average.value)

        actual_X = embeddings[:, :2]
        expected_X = np.array([[0.06223562, -0.00187974], [0.06143629, -0.00408216],
                               [0.03041881, 0.04794494]])
        self.assertEqual(True, np.allclose(expected_X, actual_X, atol=1e-03))
        self.assertEqual(3, actual_X.shape[0])


class TFIDFTest(unittest.TestCase):
    def test_word2vec_model_with_tfidf(self):
        documents_tokens = word_tokens()
        tfidf_model = TFIDF()
        tfidf_model.train(documents_tokens, filter_extremes=False)

        embeddings_model = EmbeddingsModelWord2Vec('test')
        actual_X = embeddings_model.apply(documents_tokens, AggregationMethod.average.value,
                                          tfidf_model)

        expected_X = np.array([[0.01139571, 0.00762929], [-0.0161487, -0.02562849]])
        self.assertEqual(True, np.allclose(expected_X, actual_X, atol=1e-03))

    def test_tfidf_model_calculate_document_weights(self):
        documents_tokens = word_tokens_repeated_words()
        tfidf_model = TFIDF()
        tfidf_model.train(documents_tokens, filter_extremes=False)
        documents_weights_expected = {
            'Nintendo': 0.4472135954999579,
            'games': 0.4472135954999579,
            'has': 0.4472135954999579,
            'of': 0.4472135954999579,
            'that': 0.4472135954999579
        }
        documents_weights_actual = tfidf_model.calculate_document_weights(documents_tokens[0])
        for token_name in documents_weights_expected:
            self.assertEqual(documents_weights_expected[token_name],
                             documents_weights_actual[token_name])
        self.assertEqual(len(documents_weights_expected), len(documents_weights_actual))


if __name__ == '__main__':
    unittest.main()
