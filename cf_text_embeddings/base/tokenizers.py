from nltk.data import load
from nltk.tokenize import ToktokTokenizer


def toktok_tokenizer(documents):
    tokenizer = ToktokTokenizer()
    documents_tokens = []
    for document in documents:
        document_tokens = tokenizer.tokenize(document)
        documents_tokens.append(document_tokens)
    return documents_tokens


def punkt_tokenizer(documents, language):
    supported_languages = {'english', 'estonian', 'german', 'slovene', 'spanish'}
    assert language in supported_languages, ('Language %s is not supported by punkt tokenizer' %
                                             language)

    tokenizer = load('tokenizers/punkt/{0}.pickle'.format(language))

    documents_tokens = []
    for document in documents:
        document_tokens = tokenizer.tokenize(document)
        documents_tokens.append(document_tokens)
    return documents_tokens


def token_filtering(documents_tokens, filter_tokens=None):
    default_filter = {
        '(', ')', '[', ']', '{', '}', "'", '"', ';', ':', ',', '!', '/', '\\', '@', '#', '$', '^',
        '*', '+', '=', '-', '|', '_', '<', '>', '?', '!', '.', ',', '&', '%', '0', '1', '2', '3',
        '4', '5', '6', '7', '8', '9'
    }
    filter_ = default_filter if not filter_tokens else set(
        [tok.strip() for tok in filter_tokens.split('\n')])

    documents_tokens_filtered = []
    for document_tokens in documents_tokens:
        document_tokens_filtered = [
            document_token for document_token in document_tokens if document_token not in filter_
        ]
        documents_tokens_filtered.append(document_tokens_filtered)
    return documents_tokens_filtered
