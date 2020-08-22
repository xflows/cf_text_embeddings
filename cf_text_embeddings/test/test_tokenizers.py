import unittest

from ..base.tokenizers import (punkt_tokenizer, regex_word_tokenizer,
                               token_filtering, toktok_tokenizer)


class TestTokenizers(unittest.TestCase):
    def test_toktok_tokenizer(self):
        texts = ['this is some text', 'this is some other text']
        texts_tokenized = toktok_tokenizer(texts)
        for i, text in enumerate(texts):
            tokens = text.split(' ')
            for j, token in enumerate(tokens):
                self.assertEqual(token, texts_tokenized[i][j])

    def test_regex_word_tokenizer(self):
        texts = ['this is some text', 'this is some other text']
        texts_tokenized = regex_word_tokenizer(texts)
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


class TestTokenFiltering(unittest.TestCase):
    def test_token_filtering(self):
        documents_tokens = [['this', 'is', 'some', 'text'], ['this', 'is', 'other', 'text']]
        documents_tokens_filtered = token_filtering(documents_tokens, filter_tokens=None)

        for i, _ in enumerate(documents_tokens):
            for j, _ in enumerate(documents_tokens[i]):
                self.assertEqual(documents_tokens[i][j], documents_tokens_filtered[i][j])

    def test_token_filtering_with_tokens_to_remove(self):
        documents_tokens = [['this', 'is', ',', 'some', 'text', '.']]
        documents_tokens_expected = [['this', 'is', 'some', 'text']]
        documents_tokens_actual = token_filtering(documents_tokens, filter_tokens=None)

        for i, _ in enumerate(documents_tokens_expected):
            for j, _ in enumerate(documents_tokens_expected[i]):
                self.assertEqual(documents_tokens_expected[i][j], documents_tokens_actual[i][j])

    def test_token_filtering_filter_token_empyty_string(self):
        documents_tokens = [['this', 'is', ',', 'some', 'text', '.']]
        documents_tokens_expected = [['this', 'is', 'some', 'text']]
        documents_tokens_actual = token_filtering(documents_tokens, filter_tokens='')

        for i, _ in enumerate(documents_tokens_expected):
            for j, _ in enumerate(documents_tokens_expected[i]):
                self.assertEqual(documents_tokens_expected[i][j], documents_tokens_actual[i][j])

    def test_token_filtering_with_custom_filter(self):
        documents_tokens = [['<SOS>', 'this', 'is', 'some', 'text', '<EOS>']]
        documents_tokens_expected = [['this', 'is', 'some', 'text']]
        documents_tokens_actual = token_filtering(documents_tokens, filter_tokens='<SOS>\n<EOS>')

        for i, _ in enumerate(documents_tokens_expected):
            for j, _ in enumerate(documents_tokens_expected[i]):
                self.assertEqual(documents_tokens_expected[i][j], documents_tokens_actual[i][j])

    def test_token_filtering_all_tokens_filtered(self):
        documents_tokens = [['.', ',', ')']]
        documents_tokens_expected = [[]]
        documents_tokens_actual = token_filtering(documents_tokens, filter_tokens=None)

        for i, _ in enumerate(documents_tokens_expected):
            for j, _ in enumerate(documents_tokens_expected[i]):
                self.assertEqual(documents_tokens_expected[i][j], documents_tokens_actual[i][j])


if __name__ == '__main__':
    unittest.main()
