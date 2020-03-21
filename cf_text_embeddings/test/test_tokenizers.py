import unittest

from ..base.tokenizers import punkt_tokenizer, toktok_tokenizer


class TestTokenizers(unittest.TestCase):
    def test_toktok_tokenizer(self):
        texts = ['this is some text', 'this is some other text']
        texts_tokenized = toktok_tokenizer(texts)
        for i, text in enumerate(texts):
            tokens = text.split(' ')
            for j, token in enumerate(tokens):
                self.assertEqual(token, texts_tokenized[i][j])

    def test_toktok_tokenizer_empty_input(self):
        texts = ['']
        texts_tokenized = toktok_tokenizer(texts)
        self.assertEqual(1, len(texts_tokenized))
        self.assertEqual(0, len(texts_tokenized[0]))

    def test_punkt_tokenizer(self):
        texts = ['this is some text', 'this is some other text']
        texts_tokenized = punkt_tokenizer(texts, language='english')
        for i, text in enumerate(texts):
            self.assertEqual([text], texts_tokenized[i])

    def test_punkt_tokenizer_empty_input(self):
        texts = ['']
        texts_tokenized = punkt_tokenizer(texts, language='english')
        self.assertEqual(1, len(texts_tokenized))
        self.assertEqual(0, len(texts_tokenized[0]))

if __name__ == '__main__':
    unittest.main()
