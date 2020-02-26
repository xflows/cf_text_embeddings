import unittest
import warnings
from os import path

import transformers

import numpy as np
from cf_text_embeddings.base.common import PROJECT_DATA_DIR
from cf_text_embeddings.base.embeddings_model import (AggregationMethod,
                                                      EmbeddingsModelBert,
                                                      EmbeddingsModelElmo,
                                                      EmbeddingsModelLSI,
                                                      EmbeddingsModelWord2Vec)


def word_tokens():
    return [[
        'Nintendo', 'has', 'announced', 'that', 'two', 'of', 'its', 'Switch', 'games', 'will', 'be',
        'updated.'
    ], ['Although', 'there’s', 'a', 'fierce', 'rivalry', 'between', 'smartphone', 'manufacturers.']]


def sentence_tokens():
    return [['This is sentence1 of doc1', 'this is sentence2 of doc1'],
            ['This is sentence1 of doc2', 'this is sentence2 of doc2'],
            ['This is sentence1 of doc3']]


def document_tokens():
    return [
        'Nintendo has announced that two of its Switch games will be updated.',
        'Although there’s a fierce rivalry between smartphone manufacturers.'
    ]


class EmbeddingsModelTest(unittest.TestCase):
    def test_word2vec_model(self):
        text = word_tokens()
        embeddings_model = EmbeddingsModelWord2Vec('test')
        actual_X = embeddings_model.apply(text, AggregationMethod.average.value, None)

        expected_X = np.array([[0.039, 0.026], [-0.045, -0.072]])
        self.assertEqual(True, np.allclose(expected_X, actual_X, atol=1e-03))

    def test_word2vec_model_not_exists(self):
        self.assertRaises(Exception, EmbeddingsModelWord2Vec, 'not_exists')

    def test_word2vec_model_tfidf(self):
        text = word_tokens()
        embeddings_model = EmbeddingsModelWord2Vec('test')
        actual_X = embeddings_model.apply(text, AggregationMethod.average.value, 'tfidf')

        expected_X = np.array([[0.01139571, 0.00762929], [-0.0161487, -0.02562849]])
        self.assertEqual(True, np.allclose(expected_X, actual_X, atol=1e-03))

    def test_lsi_model(self):
        np.random.seed(42)
        text = word_tokens()
        embeddings_model = EmbeddingsModelLSI(2, 1)
        actual_X = embeddings_model.apply(text, AggregationMethod.average.value, None)
        expected_X = np.array([[3.464], [2.828]])
        self.assertEqual(True, np.allclose(expected_X, actual_X, atol=1e-03))

    def test_bert_model(self):
        if not path.exists(PROJECT_DATA_DIR):
            # This test is not run by Travis CI
            warnings.warn(
                'Test test_bert_model not executed, because you need to download model first')
            return

        text = document_tokens()
        model_class = transformers.BertModel
        tokenizer_class = transformers.BertTokenizer
        pretrained_weights = 'bert-base-uncased'

        embeddings_model = EmbeddingsModelBert(
            model_class=model_class,
            tokenizer_class=tokenizer_class,
            pretrained_weights=pretrained_weights,
            vector_size=768,
            max_seq=100,
        )
        embeddings = embeddings_model.apply(text,
                                            aggregation_method=AggregationMethod.average.value,
                                            weighting_method=None)
        actual_X = embeddings[:, :2]

        expected_X = np.array([[0.13997224, -0.12020198], [0.27512622, 0.04447067]])
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
        embeddings = model.apply(sentences, aggregation_method=AggregationMethod.average.value,
                                 weighting_method=None)

        actual_X = embeddings[:, :2]
        expected_X = np.array([[-0.13834494, -0.26584834], [-0.1336689, -0.24047077],
                               [-0.13042556, -0.23901139]])
        self.assertEqual(True, np.allclose(expected_X, actual_X, atol=1e-03))
        self.assertEqual(3, actual_X.shape[0])


if __name__ == '__main__':
    unittest.main()
