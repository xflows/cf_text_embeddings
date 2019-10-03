import unittest

import numpy as np

from cf_text_embeddings.embeddings_model import (AggregationMethod,
                                                 EmbeddingsModelLSI,
                                                 EmbeddingsModelWord2Vec)
from tf_core.nltoolkit.lib.textual_data_in_out import load_adc
from tf_core.nltoolkit.lib.tokenization import (nltk_simple_tokenizer,
                                                tokenizer_hub)


def create_adc():
    input_ = """
    Nintendo\t!positive\tNintendo has announced that two of its Switch games will be updated.
    Apple\t!negative\tAlthough there’s a fierce rivalry between smartphone manufacturers."""
    input_dict = {'input': input_, 'tab_separated_title': 'true', 'leading_labels': 'true'}
    return load_adc(input_dict)['adc']


def tokenize_adc(adc, tokenizer, output_annotation):
    input_dict = {
        'adc': adc,
        'tokenizer': tokenizer['tokenizer'],
        'input_annotation': 'TextBlock',
        'output_annotation': output_annotation
    }
    return tokenizer_hub(input_dict)['adc']


def create_tokenized_adc(tokenizer, output_annotation):
    adc = create_adc()
    adc = tokenize_adc(adc, tokenizer, output_annotation)
    return adc


class EmbeddingsModelTest(unittest.TestCase):
    def test_word2vec_model(self):
        output_annotation = 'Token'
        tokenizer = nltk_simple_tokenizer({'type': 'space_tokenizer'})
        adc = create_tokenized_adc(tokenizer, output_annotation)
        documents = adc.documents
        embeddings_model = EmbeddingsModelWord2Vec('test', 'word2vec_test_model.bin')
        embeddings = embeddings_model.apply(documents, output_annotation,
                                            AggregationMethod.average.value, None)

        actual_X = embeddings.X
        actual_Y = embeddings.Y
        expected_X = np.array([[0.039, 0.026], [-0.045, -0.072]])
        expected_Y = np.array([1., 0.])
        self.assertEqual(True, np.allclose(expected_X, actual_X, atol=1e-03))
        self.assertEqual(True, np.array_equal(expected_Y, actual_Y))

    def test_word2vec_with_empty_annotations(self):
        adc = create_adc()
        documents = adc.documents
        embeddings_model = EmbeddingsModelWord2Vec('test', 'word2vec_test_model.bin')
        embeddings = embeddings_model.apply(documents, 'NonExistent',
                                            AggregationMethod.average.value, None)
        actual_X = embeddings.X
        self.assertEqual(False, np.any(actual_X))

    def test_word2vec_with_empty_annotations_and_tfidf(self):
        adc = create_adc()
        documents = adc.documents
        embeddings_model = EmbeddingsModelWord2Vec('test', 'word2vec_test_model.bin')
        embeddings = embeddings_model.apply(documents, 'NonExistent',
                                            AggregationMethod.average.value, 'tfidf')
        actual_X = embeddings.X
        self.assertEqual(False, np.any(actual_X))

    def test_word2vec_model_tfidf(self):
        output_annotation = 'Token'
        tokenizer = nltk_simple_tokenizer({'type': 'space_tokenizer'})
        adc = create_tokenized_adc(tokenizer, output_annotation)
        documents = adc.documents
        embeddings_model = EmbeddingsModelWord2Vec('test', 'word2vec_test_model.bin')
        embeddings = embeddings_model.apply(documents, output_annotation,
                                            AggregationMethod.average.value, 'tfidf')

        actual_X = embeddings.X
        actual_Y = embeddings.Y
        expected_X = np.array([[0.01139571, 0.00762929], [-0.0161487, -0.02562849]])
        expected_Y = np.array([1., 0.])
        self.assertEqual(True, np.allclose(expected_X, actual_X, atol=1e-03))
        self.assertEqual(True, np.array_equal(expected_Y, actual_Y))

    def test_lsi_model(self):
        np.random.seed(42)
        output_annotation = 'Token'
        tokenizer = nltk_simple_tokenizer({'type': 'space_tokenizer'})
        adc = create_tokenized_adc(tokenizer, output_annotation)
        documents = adc.documents
        embeddings_model = EmbeddingsModelLSI(2, 1)
        embeddings = embeddings_model.apply(documents, output_annotation,
                                            AggregationMethod.average.value, None)
        actual_X = embeddings.X
        actual_Y = embeddings.Y
        expected_X = np.array([[3.464], [2.828]])
        expected_Y = np.array([1., 0.])
        self.assertEqual(True, np.allclose(expected_X, actual_X, atol=1e-03))
        self.assertEqual(True, np.array_equal(expected_Y, actual_Y))
