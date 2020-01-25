import csv


def read_file(filename, delimiter):
    bytes_to_read = 1024 * 2  # for analysis
    rows = []
    with open(filename) as csvfile:
        if delimiter is None:
            sniffer = csv.Sniffer()
            content_to_analyze = csvfile.read(bytes_to_read)
            dialect = sniffer.sniff(content_to_analyze)
            csvfile.seek(0)
            csv_reader = csv.reader(csvfile, dialect=dialect)
        else:
            csv_reader = csv.reader(csvfile, delimiter=delimiter)

        for row in csv_reader:
            if not row:
                continue
            rows.append(row)
    return rows


def parse_csv(rows, title_column, text_column, label_column):
    custom_columns = [title_column, text_column, label_column]
    predifined_columns = ['title', 'text', 'label']
    column_map = dict(zip(custom_columns, predifined_columns))
    output = {column: [] for column in predifined_columns}
    columns_indices = {}

    # set column indices
    first_row = rows[0]
    first_row_indices = dict((v, k) for k, v in enumerate(first_row))
    for column in custom_columns:
        columns_indices[column] = first_row_indices.get(column, -1)

    if columns_indices[text_column] == -1:
        raise Exception('Text column: %s is not found in text' % text_column)

    for row in rows[1:]:
        if not row:
            continue
        for column_name in custom_columns:
            column_index = columns_indices[column_name]
            if column_index == -1 or column_index >= len(row):
                output[column_map[column_name]].append('')
                continue
            column_value = row[column_index]
            output[column_map[column_name]].append(column_value)
    return output


def read_csv(filename, delimiter, title_column, text_column, label_column):
    rows = read_file(filename, delimiter)
    if not rows:
        raise Exception('The file is empty')
    document_corpus = parse_csv(rows, title_column, text_column, label_column)
    return document_corpus
