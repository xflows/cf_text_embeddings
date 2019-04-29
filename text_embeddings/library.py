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
    return model_name


def text_embeddings_word2vec(input_dict):
    languages = {
        'en': 'GoogleNews-vectors-negative300.wv.bin',
        'es': 'SBW-vectors-300-min5.wv.bin',
    }
    model_name = text_embeddings_extract_model_name(input_dict, languages)
    return {'embeddings_model': EmbeddingsModelWord2Vec(model_name)}


def text_embeddings_glove(input_dict):
    languages = {
        'en': 'glove.6B.300d.wv.bin',
    }
    model_name = text_embeddings_extract_model_name(input_dict, languages)
    return {'embeddings_model': EmbeddingsModelGloVe(model_name)}


def text_embeddings_fasttext(input_dict):
    languages = {
        # model downloaded and converted with gensim:
        # https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
        'en': 'wiki-news-300d-1M.bin',
    }
    model_name = text_embeddings_extract_model_name(input_dict, languages)
    return {'embeddings_model': EmbeddingsModelFastText(model_name)}


def text_embeddings_bert(_):
    return {
        'embeddings_model':
        EmbeddingsModelBert(model_name='bert_12_768_12', dataset_name='wiki_multilingual_cased',
                            max_seq_length=1000, default_token_annotation='Sentence')
    }


def text_embeddings_universal_sentence_encoder(input_dict):
    languages = {
        # https://tfhub.dev/google/universal-sentence-encoder/2
        'en': 'universal_sentence_encoder_english',
    }
    model_name = text_embeddings_extract_model_name(input_dict, languages)
    return {
        'embeddings_model':
        EmbeddingsModelUniversalSentenceEncoder(model_name, default_token_annotation='Sentence')
    }


def text_embeddings_doc2vec(input_dict):
    languages = {
        'en': 'doc2vec.bin',
    }
    model_name = text_embeddings_extract_model_name(input_dict, languages)
    return {
        'embeddings_model': EmbeddingsModelDoc2Vec(model_name, default_token_annotation='TextBlock')
    }


def text_embeddings_elmo(input_dict):
    languages = {
        # https://tfhub.dev/google/elmo/2
        'en': 'elmo_english',
    }
    model_name = text_embeddings_extract_model_name(input_dict, languages)
    return {
        'embeddings_model':
        EmbeddingsModelElmo(model_name, model_output='default', signature="tokens", as_dict=True)
    }


def text_embeddings_embeddings_hub(input_dict):
    embeddings_model = input_dict['embeddings_model']
    default_token_annotation = embeddings_model.default_token_annotation
    token_annotation = input_dict['token_annotation'] or default_token_annotation or 'Token'
    aggregation_method = input_dict['aggregation_method']
    adc = input_dict['adc']
    documents = adc.documents
    bow_dataset = embeddings_model.apply(documents, token_annotation, aggregation_method)
    return {'bow_dataset': bow_dataset}


def text_embeddings_language(input_dict):
    return input_dict
