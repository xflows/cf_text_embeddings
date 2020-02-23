import transformers
from Orange.data import Table

from cf_text_embeddings.base import io, tokenizers
from cf_text_embeddings.base.common import (load_numpy_array, map_checkbox_value, map_y, to_float,
                                            to_int)
from cf_text_embeddings.base.embeddings_model import (EmbeddingsModelBert, EmbeddingsModelDoc2Vec,
                                                      EmbeddingsModelElmo, EmbeddingsModelFastText,
                                                      EmbeddingsModelGloVe, EmbeddingsModelLSI,
                                                      EmbeddingsModelUniversalSentenceEncoder,
                                                      EmbeddingsModelWord2Vec)
from cf_text_embeddings.base.table import orange_data_table


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
    assert 'text' in input_dict, 'Text is missing'

    language = input_dict['language']
    input_dict['texts'] = tokenizers.punkt_tokenizer(input_dict['texts'], language)
    return input_dict


def cf_text_embeddings_base(klass, input_dict):
    lang = input_dict.get('lang_selector') or input_dict.get('lang')
    embeddings_model = klass(lang)

    aggregation_method = input_dict.get('aggregation_method')
    weighting_method = input_dict.get('weighting_method')
    texts = input_dict['texts']
    embeddings = embeddings_model.apply(texts, aggregation_method, weighting_method)

    labels = input_dict['labels']
    dataset = orange_data_table(embeddings, labels)
    return {'dataset': dataset}


def cf_text_embeddings_word2vec(input_dict):
    return cf_text_embeddings_base(EmbeddingsModelWord2Vec, input_dict)


def cf_text_embeddings_glove(input_dict):
    return cf_text_embeddings_base(EmbeddingsModelGloVe, input_dict)


def cf_text_embeddings_fasttext(input_dict):
    languages = {
        # https://github.com/facebookresearch/MUSE
        'en': 'wiki.multi.en.wv',
        # https://github.com/facebookresearch/MUSE
        'es': 'wiki.multi.es.wv',
        # https://github.com/facebookresearch/MUSE
        'de': 'wiki.multi.de.wv',
        # https://github.com/facebookresearch/MUSE
        'ru': 'wiki.multi.ru.wv',
        # https://fasttext.cc/docs/en/crawl-vectors.html
        'lv': 'fasttext_lv.wv',
        # https://fasttext.cc/docs/en/crawl-vectors.html
        'lt': 'fasttext_lt.wv',
        # https://github.com/facebookresearch/MUSE
        'ee': 'wiki.multi.ee.wv',
        # https://github.com/facebookresearch/MUSE
        'sl': 'wiki.multi.sl.wv',
        # https://github.com/facebookresearch/MUSE
        'hr': 'wiki.multi.hr.wv',
    }
    lang, model_name = cf_text_embeddings_extract_models_language(input_dict, languages)
    return {'embeddings_model': EmbeddingsModelFastText(lang, model_name)}


def cf_text_embeddings_lsi(input_dict):
    num_topics_default = 200
    num_topics = to_int(input_dict['num_topics'], num_topics_default)
    num_topics = num_topics_default if num_topics < 1 else num_topics
    decay = to_float(input_dict['decay'], 1.0)
    return {'embeddings_model': EmbeddingsModelLSI(num_topics, decay)}


def cf_text_embeddings_bert(_):
    model_class = transformers.BertModel
    tokenizer_class = transformers.BertTokenizer
    pretrained_weights = 'bert-base-uncased'
    return {
        # Model source: https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1
        'embeddings_model':
        EmbeddingsModelBert(model_class=model_class, tokenizer_class=tokenizer_class,
                            vector_size=768, pretrained_weights=pretrained_weights, max_seq=100,
                            default_token_annotation='TextBlock')
    }


def cf_text_embeddings_universal_sentence_encoder(input_dict):
    languages = {
        # https://tfhub.dev/google/universal-sentence-encoder/2
        'en': 'universal_sentence_encoder_english',
        # https://tfhub.dev/google/universal-sentence-encoder-xling/en-de/1
        'de': 'universal_sentence_encoder_german',
        # https://tfhub.dev/google/universal-sentence-encoder-xling/en-es/1
        'es': 'universal_sentence_encoder_spanish',
    }
    lang, model_name = cf_text_embeddings_extract_models_language(input_dict, languages)

    return {
        'embeddings_model':
        EmbeddingsModelUniversalSentenceEncoder(lang, model_name,
                                                default_token_annotation='Sentence')
    }


def cf_text_embeddings_doc2vec(input_dict):
    languages = {
        # https://github.com/jhlau/doc2vec
        'en': 'doc2vec.bin',
    }
    lang, model_name = cf_text_embeddings_extract_models_language(input_dict, languages)

    return {
        'embeddings_model':
        EmbeddingsModelDoc2Vec(lang, model_name, default_token_annotation='TextBlock')
    }


def cf_text_embeddings_elmo(input_dict):
    languages = {
        # http://vectors.nlpl.eu/repository/#
        'en': 'elmo',
        # http://vectors.nlpl.eu/repository/#
        'sl': 'elmo',
        # http://vectors.nlpl.eu/repository/#
        'es': 'elmo',
        # http://vectors.nlpl.eu/repository/#
        'ru': 'elmo',
        # http://vectors.nlpl.eu/repository/#
        'hr': 'elmo',
        # http://vectors.nlpl.eu/repository/#
        'ee': 'elmo',
        # http://vectors.nlpl.eu/repository/#
        'lv': 'elmo',
        # http://vectors.nlpl.eu/repository/#
        'de': 'elmo',
    }
    lang, model_name = cf_text_embeddings_extract_models_language(input_dict, languages)

    return {'embeddings_model': EmbeddingsModelElmo(lang, model_name)}


def cf_text_embeddings_language(input_dict):
    return input_dict


def cf_text_embeddings_import_dataset(input_dict):
    x_path = input_dict['x_path']
    y_path = input_dict['y_path']

    x = load_numpy_array(x_path)
    y = load_numpy_array(y_path)

    y, unique_labels = map_y(y)
    domain = orange_domain(x.shape[1], unique_labels)
    bow_dataset = Table(domain, x, Y=y)
    return {'bow_dataset': bow_dataset}


def cf_text_embeddings_export_dataset(input_dict):
    return input_dict
