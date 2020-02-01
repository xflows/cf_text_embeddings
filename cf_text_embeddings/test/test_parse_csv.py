import unittest

from cf_text_embeddings.base.io import parse_csv, validate_and_convert_indices


class TestParseCSV(unittest.TestCase):
    def test_parse_csv(self):
        rows = [['title', 'text', 'label'], ['doc1', 'This is some text', '1'],
                ['doc2', 'This is some other text', '0']]
        dc = parse_csv(rows, 1, 2, 3)

        for i, column in enumerate(['title', 'text', 'label']):
            self.assertEqual(len(rows), len(dc[column]))
        for i, row in enumerate(rows):
            for j, column in enumerate(['title', 'text', 'label']):
                self.assertEqual(row[j], dc[column][i])

    def test_multiple_spaces_in_text(self):
        rows = [['title', 'text', 'label'], ['doc1', 'This is    some text ', '1'],
                ['doc2', 'This is some      other text', '0']]
        dc = parse_csv(rows, 1, 2, 3)

        for column in ['title', 'text', 'label']:
            self.assertEqual(len(rows), len(dc[column]))
        for i, row in enumerate(rows):
            for j, column in enumerate(['title', 'text', 'label']):
                self.assertEqual(row[j], dc[column][i])

    def test_no_title(self):
        rows = [['text', 'label'], ['This is some text', '1'], ['This is some other text', '0']]
        dc = parse_csv(rows, None, 1, 2)

        for i, column in enumerate(['text', 'label']):
            self.assertEqual(len(rows), len(dc[column]))

        self.assertEqual(dc['title'], [])
        for i, row in enumerate(rows):
            for j, column in enumerate(['text', 'label']):
                self.assertEqual(row[j], dc[column][i])

    def test_no_label(self):
        rows = [['title', 'text'], ['doc1', 'This is some text'],
                ['doc2', 'This is some other text']]
        dc = parse_csv(rows, 1, 2, None)

        self.assertEqual(dc['label'], [])
        for i, column in enumerate(['title', 'text']):
            self.assertEqual(len(rows), len(dc[column]))
        for i, row in enumerate(rows):
            for j, column in enumerate(['title', 'text']):
                self.assertEqual(row[j], dc[column][i])

    def test_no_title_no_label(self):
        rows = [['text'], ['This is some text'], ['This is some other text']]
        dc = parse_csv(rows, None, 1, None)

        self.assertEqual(dc['title'], [])
        self.assertEqual(dc['label'], [])
        for i, column in enumerate(['text']):
            self.assertEqual(len(rows), len(dc[column]))
        for i, row in enumerate(rows):
            for j, column in enumerate(['text']):
                self.assertEqual(row[j], dc[column][i])


class ValidateAndConvertIndices(unittest.TestCase):
    def test_all_indices(self):
        title_index, text_index, label_index = validate_and_convert_indices(1, 2, 3)
        self.assertEqual(title_index, 1)
        self.assertEqual(text_index, 2)
        self.assertEqual(label_index, 3)

    def test_all_indices_string(self):
        title_index, text_index, label_index = validate_and_convert_indices('1', '2', '3')
        self.assertEqual(title_index, 1)
        self.assertEqual(text_index, 2)
        self.assertEqual(label_index, 3)

    def test_zero_index(self):
        self.assertRaises(Exception, validate_and_convert_indices, '0', '1', '')

    def test_title_label_empty_string(self):
        title_index, text_index, label_index = validate_and_convert_indices('', '1', '')
        self.assertEqual(title_index, None)
        self.assertEqual(text_index, 1)
        self.assertEqual(label_index, None)

    def test_title_label_letters(self):
        title_index, text_index, label_index = validate_and_convert_indices('a', '1', 'b')
        self.assertEqual(title_index, None)
        self.assertEqual(text_index, 1)
        self.assertEqual(label_index, None)

    def test_title_label_none(self):
        title_index, text_index, label_index = validate_and_convert_indices(None, '1', None)
        self.assertEqual(title_index, None)
        self.assertEqual(text_index, 1)
        self.assertEqual(label_index, None)

    def test_no_text(self):
        self.assertRaises(Exception, validate_and_convert_indices, '', '', '')


if __name__ == '__main__':
    unittest.main()
