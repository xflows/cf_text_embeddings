from text_embeddings.embeddings_model import (
    EmbeddingsModelBert, EmbeddingsModelDoc2Vec, EmbeddingsModelElmo,
    EmbeddingsModelFastText, EmbeddingsModelGloVe,
    EmbeddingsModelUniversalSentenceEncoder, EmbeddingsModelWord2Vec)


def text_embeddings_extract_model_name(input_dict, languages):
    # language selector overrides the language in the widget
    lang = input_dict['lang_selector'] or input_dict['lang']
    model_name = languages.get(lang)
    if model_name is None:
        raise Exception('Model for %s language is not supported' % lang)
    return lang, model_name


def text_embeddings_word2vec(input_dict):
    languages = {
        # https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM
        'en': 'GoogleNews-vectors-negative300.wv.bin',
        # https://github.com/uchile-nlp/spanish-word-embeddings
        'es': 'SBW-vectors-300-min5.wv.bin',
        # http://vectors.nlpl.eu/repository/
        'sl': 'word2vec_si.wv',
        # http://vectors.nlpl.eu/repository/
        'hr': 'word2vec_hr.wv',
        # http://vectors.nlpl.eu/repository/
        'de': 'word2vec_de.wv',
        # http://vectors.nlpl.eu/repository/
        'ru': 'word2vec_ru.wv',
        # http://vectors.nlpl.eu/repository/
        'lv': 'word2vec_lv.wv',
        # http://vectors.nlpl.eu/repository/
        'ee': 'word2vec_ee.wv',
    }
    lang, model_name = text_embeddings_extract_model_name(input_dict, languages)
    return {'embeddings_model': EmbeddingsModelWord2Vec(lang, model_name)}


def text_embeddings_glove(input_dict):
    languages = {
        # https://nlp.stanford.edu/projects/glove/
        'en': 'glove.6B.300d.wv.bin',
        # https://github.com/dccuchile/spanish-word-embeddings
        'es': 'glove_es.wv',
        # https://deepset.ai/german-word-embeddings
        'de': 'glove_de.wv',
    }
    lang, model_name = text_embeddings_extract_model_name(input_dict, languages)
    return {'embeddings_model': EmbeddingsModelGloVe(lang, model_name)}


def text_embeddings_fasttext(input_dict):
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
    lang, model_name = text_embeddings_extract_model_name(input_dict, languages)
    return {'embeddings_model': EmbeddingsModelFastText(lang, model_name)}


def text_embeddings_bert(_):
    return {
        # Model source: https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1
        'embeddings_model':
        EmbeddingsModelBert(model_name='bert_12_768_12', dataset_name='wiki_multilingual_cased',
                            max_seq_length=1000, default_token_annotation='Sentence')
    }


def text_embeddings_universal_sentence_encoder(input_dict):
    languages = {
        # https://tfhub.dev/google/universal-sentence-encoder/2
        'en': 'universal_sentence_encoder_english',
        # https://tfhub.dev/google/universal-sentence-encoder-xling/en-de/1
        'de': 'universal_sentence_encoder_german',
        # https://tfhub.dev/google/universal-sentence-encoder-xling/en-es/1
        'es': 'universal_sentence_encoder_spanish',
    }
    lang, model_name = text_embeddings_extract_model_name(input_dict, languages)
    return {
        'embeddings_model':
        EmbeddingsModelUniversalSentenceEncoder(lang, model_name,
                                                default_token_annotation='Sentence')
    }


def text_embeddings_doc2vec(input_dict):
    languages = {
        # https://github.com/jhlau/doc2vec
        'en': 'doc2vec.bin',
    }
    lang, model_name = text_embeddings_extract_model_name(input_dict, languages)
    return {
        'embeddings_model':
        EmbeddingsModelDoc2Vec(lang, model_name, default_token_annotation='TextBlock')
    }


def text_embeddings_elmo(input_dict):
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
    lang, model_name = text_embeddings_extract_model_name(input_dict, languages)
    return {'embeddings_model': EmbeddingsModelElmo(lang, model_name)}


def text_embeddings_embeddings_hub(input_dict):
    embeddings_model = input_dict['embeddings_model']

    default_token_annotation = embeddings_model.default_token_annotation
    token_annotation = input_dict['token_annotation'] or default_token_annotation or 'Token'
    aggregation_method = input_dict['aggregation_method']
    weighting_method = input_dict['weighting_method']
    adc = input_dict['adc']
    documents = adc.documents
    bow_dataset = embeddings_model.apply(documents, token_annotation, aggregation_method,
                                         weighting_method)
    return {'bow_dataset': bow_dataset}


def text_embeddings_language(input_dict):
    return input_dict
