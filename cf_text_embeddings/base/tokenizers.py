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
