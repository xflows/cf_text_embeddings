import csv


def map_delimiter(delimiter):
    if delimiter == '\\t':
        return '\t'
    return delimiter


def read_file(filename, delimiter, skip_header):
    delimiter = map_delimiter(delimiter)
    bytes_to_read = 1024 * 10  # for analysis
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


def parse_csv(rows, text_index, label_index):
    texts, labels = [], []
    for row in rows:
        if not row:
            continue
        if text_index and 0 <= text_index - 1 < len(row):
            texts.append(row[text_index - 1])
        if label_index and 0 <= label_index - 1 < len(row):
            labels.append(row[label_index - 1])
    return texts, labels


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


def validate_and_convert_indices(text_index, label_index):
    text_index = validate_and_convert_index(text_index)
    label_index = validate_and_convert_index(label_index)
    if text_index is None:
        raise Exception('Text index is not set correctly')
    return text_index, label_index


def read_csv(filename, delimiter, skip_header, text_index, label_index):
    text_index, label_index = validate_and_convert_indices(text_index, label_index)
    rows = read_file(filename, delimiter, skip_header)
    if not rows:
        raise Exception('Cannot read the csv')
    texts, labels = parse_csv(rows, text_index, label_index)

    if not texts:
        raise Exception('Text could not be read from a file')
    return texts, labels
