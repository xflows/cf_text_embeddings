from text_embeddings.embeddings_model import (EmbeddingsModelBert,
                                              EmbeddingsModelDoc2Vec,
                                              EmbeddingsModelElmo,
                                              EmbeddingsModelFastText,
                                              EmbeddingsModelGloVe,
                                              EmbeddingsModelUniversalSentenceEncoder,
                                              EmbeddingsModelWord2Vec)


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
    }
    lang, model_name = text_embeddings_extract_model_name(input_dict, languages)
    return {'embeddings_model': EmbeddingsModelWord2Vec(lang, model_name)}


def text_embeddings_glove(input_dict):
    languages = {
        # https://nlp.stanford.edu/projects/glove/
        'en': 'glove.6B.300d.wv.bin',
    }
    lang, model_name = text_embeddings_extract_model_name(input_dict, languages)
    return {'embeddings_model': EmbeddingsModelGloVe(lang, model_name)}


def text_embeddings_fasttext(input_dict):
    languages = {
        # https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
        'en': 'wiki-news-300d-1M.bin',
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
        # https://tfhub.dev/google/elmo/2
        'en': 'elmo_english',
    }
    lang, model_name = text_embeddings_extract_model_name(input_dict, languages)
    return {
        'embeddings_model':
        EmbeddingsModelElmo(lang, model_name, model_output='default', signature="tokens",
                            as_dict=True)
    }


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
