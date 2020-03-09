import unittest

from ..base.io import parse_csv, validate_and_convert_indices


class TestParseCSV(unittest.TestCase):
    def test_parse_csv(self):
        rows = [['text', 'label'], ['This is some text', '1'], ['This is some other text', '0']]
        texts, labels = parse_csv(rows, 1, 2)
        dc = {'text': texts, 'label': labels}

        for i, column in enumerate(['text', 'label']):
            self.assertEqual(len(rows), len(dc[column]))
        for i, row in enumerate(rows):
            for j, column in enumerate(['text', 'label']):
                self.assertEqual(row[j], dc[column][i])

    def test_multiple_spaces_in_text(self):
        rows = [['text', 'label'], ['This is    some text ', '1'],
                ['This is some      other text', '0']]
        texts, labels = parse_csv(rows, 1, 2)
        dc = {'text': texts, 'label': labels}

        for column in ['text', 'label']:
            self.assertEqual(len(rows), len(dc[column]))
        for i, row in enumerate(rows):
            for j, column in enumerate(['text', 'label']):
                self.assertEqual(row[j], dc[column][i])

    def test_no_label(self):
        rows = [['text'], ['This is some text'], ['This is some other text']]
        texts, labels = parse_csv(rows, 1, None)
        dc = {'text': texts, 'label': labels}

        self.assertEqual(dc['label'], [])
        for i, column in enumerate(['text']):
            self.assertEqual(len(rows), len(dc[column]))
        for i, row in enumerate(rows):
            for j, column in enumerate(['text']):
                self.assertEqual(row[j], dc[column][i])


class ValidateAndConvertIndices(unittest.TestCase):
    def test_all_indices(self):
        text_index, label_index = validate_and_convert_indices(1, 2)
        self.assertEqual(text_index, 1)
        self.assertEqual(label_index, 2)

    def test_all_indices_string(self):
        text_index, label_index = validate_and_convert_indices('1', '2')
        self.assertEqual(text_index, 1)
        self.assertEqual(label_index, 2)

    def test_zero_index(self):
        self.assertRaises(Exception, validate_and_convert_indices, '0', '1')

    def test_label_empty_string(self):
        text_index, label_index = validate_and_convert_indices('1', '')
        self.assertEqual(text_index, 1)
        self.assertEqual(label_index, None)

    def test_label_letters(self):
        text_index, label_index = validate_and_convert_indices('1', 'b')
        self.assertEqual(text_index, 1)
        self.assertEqual(label_index, None)

    def test_no_text(self):
        self.assertRaises(Exception, validate_and_convert_indices, '', '')


if __name__ == '__main__':
    unittest.main()
