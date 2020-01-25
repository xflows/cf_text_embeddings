import unittest

from cf_text_embeddings.base.io import parse_csv


class TestParseCSV(unittest.TestCase):
    def test_parse_csv(self):
        rows = [['title', 'text', 'label'], ['doc1', 'This is some text', '1'],
                ['doc2', 'This is some other text', '0']]
        dc = parse_csv(rows, 'title', 'text', 'label')

        for i, column in enumerate(['title', 'text', 'label']):
            self.assertEqual(len(rows) - 1, len(dc[column]))
        for i, row in enumerate(rows[1:]):
            for j, column in enumerate(['title', 'text', 'label']):
                self.assertEqual(row[j], dc[column][i])

    def test_custom_property_names(self):
        rows = [['some special title', 'some special text', 'some special label'],
                ['doc1', 'This is some text', '1'], ['doc2', 'This is some other text', '0']]
        dc = parse_csv(rows, 'some special title', 'some special text', 'some special label')

        for column in ['title', 'text', 'label']:
            self.assertEqual(len(rows) - 1, len(dc[column]))
        for i, row in enumerate(rows[1:]):
            for j, column in enumerate(['title', 'text', 'label']):
                self.assertEqual(row[j], dc[column][i])

    def test_multiple_spaces_in_text(self):
        rows = [['title', 'text', 'label'], ['doc1', 'This is    some text ', '1'],
                ['doc2', 'This is some      other text', '0']]
        dc = parse_csv(rows, 'title', 'text', 'label')

        for column in ['title', 'text', 'label']:
            self.assertEqual(len(rows) - 1, len(dc[column]))
        for i, row in enumerate(rows[1:]):
            for j, column in enumerate(['title', 'text', 'label']):
                self.assertEqual(row[j], dc[column][i])

    def test_no_title(self):
        rows = [['text', 'label'], ['This is some text', '1'], ['This is some other text', '0']]
        dc = parse_csv(rows, '', 'text', 'label')

        for i, column in enumerate(['title', 'text', 'label']):
            self.assertEqual(len(rows) - 1, len(dc[column]))

        for i, row in enumerate(rows[1:]):
            for j, column in enumerate(['text', 'label']):
                self.assertEqual(row[j], dc[column][i])

    def test_no_label(self):
        rows = [['title', 'text'], ['doc1', 'This is some text'],
                ['doc2', 'This is some other text']]
        dc = parse_csv(rows, 'title', 'text', '')

        for i, column in enumerate(['title', 'text', 'label']):
            self.assertEqual(len(rows) - 1, len(dc[column]))
        for i, row in enumerate(rows[1:]):
            for j, column in enumerate(['title', 'text']):
                self.assertEqual(row[j], dc[column][i])


if __name__ == '__main__':
    unittest.main()
