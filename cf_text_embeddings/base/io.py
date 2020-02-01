import csv


def map_delimiter(delimiter):
    if delimiter == '\\t':
        return '\t'
    return delimiter


def read_file(filename, delimiter, skip_header):
    delimiter = map_delimiter(delimiter)
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
    if skip_header:
        rows = rows[1:]
    return rows


def parse_csv(rows, title_index, text_index, label_index):
    predifined_columns = ['title', 'text', 'label']
    columns_indexes = list(zip(predifined_columns, [title_index, text_index, label_index]))
    output = {column: [] for column in predifined_columns}

    for row in rows:
        if not row:
            continue
        for column, index in columns_indexes:
            if index and index <= len(row):
                output[column].append(row[index - 1])
    return output


def validate_index(index):
    if not index:
        return False
    if isinstance(index, str) and not index.isdigit():
        return False
    index = int(index)
    if index < 1:
        raise Exception('Index should be greater or equal to 1')
    return True


def convert_to_int(value):
    return int(value)


def validate_and_convert_index(index):
    return convert_to_int(index) if validate_index(index) else None


def validate_and_convert_indices(title_index, text_index, label_index):
    title_index = validate_and_convert_index(title_index)
    text_index = validate_and_convert_index(text_index)
    label_index = validate_and_convert_index(label_index)
    if text_index is None:
        raise Exception('Text index is not set correctly')
    return title_index, text_index, label_index


def read_csv(filename, delimiter, skip_header, title_index, text_index, label_index):
    title_index, text_index, label_index = validate_and_convert_indices(
        title_index, text_index, label_index)
    rows = read_file(filename, delimiter, skip_header)
    if not rows:
        raise Exception('Cannot read the csv')
    document_corpus = parse_csv(rows, title_index, text_index, label_index)

    if not document_corpus['text']:
        raise Exception('Text could not be read from a file')
    return document_corpus
