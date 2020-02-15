import unittest
from os import path

from cf_text_embeddings.base.common import cf_text_embeddings_package_path
from cf_text_embeddings.base.embeddings_model import AggregationMethod
from cf_text_embeddings.library import (cf_text_embeddings_parse_csv,
                                        cf_text_embeddings_tok_tok_tokenizer,
                                        cf_text_embeddings_word2vec)


class TestLibrary(unittest.TestCase):
    def test_parse_csv(self):
        package_path = cf_text_embeddings_package_path()
        input_dict = {
            'input': path.join(package_path, 'corpus/basic.tsv'),
            'text_index': 2,
            'label_index': 3,
            'skip_header': 'true',
        }
        output_dict = cf_text_embeddings_parse_csv(input_dict)
        self.assertEqual(True, 'texts' in output_dict)
        self.assertEqual(True, 'labels' in output_dict)
        self.assertEqual(True, len(output_dict) == 2)
        self.assertEqual(True, len(output_dict['texts']) == 4)
        self.assertEqual(True, len(output_dict['labels']) == 4)

    def test_toktok_tokenizer(self):
        input_dict = {'texts': ['this is doc1'], 'labels': [0]}
        output_dict = cf_text_embeddings_tok_tok_tokenizer(input_dict)
        self.assertEqual(True, 'texts' in output_dict)
        self.assertEqual(True, 'labels' in output_dict)
        self.assertEqual(True, len(output_dict) == 2)
        self.assertEqual(True, len(output_dict['texts']) == 1)
        self.assertEqual(True, len(output_dict['texts'][0]) == 3)
        self.assertEqual(True, len(output_dict['labels']) == 1)

    def test_word2vec_model(self):
        input_dict = {
            'texts': [['this', 'is', 'doc1'], ['this', 'is', 'doc2']],
            'labels': ['positive', 'negative'],
            'lang': 'en',
            'aggregation_method': AggregationMethod.summation.value,
            'weighting_method': None,
        }
        output_dict = cf_text_embeddings_word2vec(input_dict)
        self.assertTrue('dataset' in output_dict)
        self.assertTrue(len(output_dict['dataset']) == 2)


if __name__ == '__main__':
    unittest.main()
