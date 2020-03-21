from .base import io, tokenizers
from .base.common import load_numpy_array, map_checkbox_value, to_float, to_int
from .base.embeddings_model import (TFIDF, EmbeddingsModelBert,
                                    EmbeddingsModelDoc2Vec,
                                    EmbeddingsModelElmo,
                                    EmbeddingsModelFastText,
                                    EmbeddingsModelGloVe, EmbeddingsModelLSI,
                                    EmbeddingsModelUniversalSentenceEncoder,
                                    EmbeddingsModelWord2Vec)
from .base.table import orange_data_table


def cf_text_embeddings_parse_csv(input_dict):
    filename = input_dict['input']
    text_index = input_dict.get('text_index')
    label_index = input_dict.get('label_index')
    delimiter = input_dict.get('delimiter') or None
    skip_header = input_dict.get('skip_header')
    skip_header = map_checkbox_value(skip_header)

    texts, labels = io.read_csv(filename=filename, delimiter=delimiter, skip_header=skip_header,
                                text_index=text_index, label_index=label_index)
    return {'texts': texts, 'labels': labels}


def cf_text_embeddings_tok_tok_tokenizer(input_dict):
    assert 'texts' in input_dict, 'Text is missing'

    input_dict['texts'] = tokenizers.toktok_tokenizer(input_dict['texts'])
    return input_dict


def cf_text_embeddings_punkt_tokenizer(input_dict):
    assert 'texts' in input_dict, 'Text is missing'

    language = input_dict['language']
    input_dict['texts'] = tokenizers.punkt_tokenizer(input_dict['texts'], language)
    return input_dict


def cf_text_embeddings_token_filtering(input_dict):
    assert 'texts' in input_dict, 'Text is missing'

    input_dict['texts'] = tokenizers.token_filtering(input_dict['texts'],
                                                     filter_tokens=input_dict['filter_tokens'])
    return input_dict


def cf_text_embeddings_tfidf(input_dict):
    assert 'texts' in input_dict, 'Text is missing'

    tfidf = TFIDF()
    tfidf.train(input_dict['texts'])
    return {'tfidf': tfidf}


def cf_text_embeddings_base(klass, input_dict):
    lang = input_dict.get('lang_selector') or input_dict.get('lang')
    embeddings_model = klass(lang)

    aggregation_method = input_dict.get('aggregation_method')
    tfidf = input_dict.get('tfidf')
    texts = input_dict['texts']
    embeddings = embeddings_model.apply(texts, aggregation_method, tfidf=tfidf)
    return {'embeddings': embeddings}


def cf_text_embeddings_word2vec(input_dict):
    return cf_text_embeddings_base(EmbeddingsModelWord2Vec, input_dict)


def cf_text_embeddings_glove(input_dict):
    return cf_text_embeddings_base(EmbeddingsModelGloVe, input_dict)


def cf_text_embeddings_fasttext(input_dict):
    return cf_text_embeddings_base(EmbeddingsModelFastText, input_dict)


def cf_text_embeddings_elmo(input_dict):
    return cf_text_embeddings_base(EmbeddingsModelElmo, input_dict)


def cf_text_embeddings_lsi(input_dict):
    texts = input_dict['texts']
    num_topics_default = 200
    num_topics = to_int(input_dict['num_topics'], num_topics_default)
    num_topics = num_topics_default if num_topics < 1 else num_topics
    decay = to_float(input_dict['decay'], 1.0)

    embeddings_model = EmbeddingsModelLSI(num_topics, decay)
    embeddings = embeddings_model.apply(texts)
    return {'embeddings': embeddings}


def cf_text_embeddings_bert(input_dict):
    texts = input_dict['texts']

    embeddings_model = EmbeddingsModelBert()
    embeddings = embeddings_model.apply(texts)
    return {'embeddings': embeddings}


def cf_text_embeddings_universal_sentence_encoder(input_dict):
    return cf_text_embeddings_base(EmbeddingsModelUniversalSentenceEncoder, input_dict)


def cf_text_embeddings_doc2vec(input_dict):
    lang = input_dict.get('lang_selector') or input_dict.get('lang')
    texts = input_dict['texts']

    embeddings_model = EmbeddingsModelDoc2Vec(lang)
    embeddings = embeddings_model.apply(texts)
    return {'embeddings': embeddings}


def cf_text_embeddings_language(input_dict):
    return input_dict


def cf_text_embeddings_import_dataset(input_dict):
    x_path = input_dict['x_path']
    y_path = input_dict['y_path']

    embeddings = load_numpy_array(x_path)
    labels = load_numpy_array(y_path)
    dataset = orange_data_table(embeddings, labels)
    return {'dataset': dataset}


def cf_text_embeddings_create_dataset(input_dict):
    embeddings = input_dict['embeddings']
    labels = input_dict['labels']
    dataset = orange_data_table(embeddings, labels)
    return {'dataset': dataset}


def cf_text_embeddings_export_dataset(input_dict):
    return input_dict
