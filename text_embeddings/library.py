from text_embeddings.embeddings_model import (EmbeddingsModelDoc2Vec,
                                              EmbeddingsModelElmo,
                                              EmbeddingsModelFastText,
                                              EmbeddingsModelGloVe,
                                              EmbeddingsModelUniversalSentenceEncoder,
                                              EmbeddingsModelWord2Vec)


def text_embeddings_extract_model_name(input_dict, languages):
    lang = input_dict['lang']
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


def text_embeddings_universal_sentence_encoder(input_dict):
    languages = {
        # https://tfhub.dev/google/universal-sentence-encoder/2
        'en': 'universal_sentence_encoder_english',
    }
    model_name = text_embeddings_extract_model_name(input_dict, languages)
    return {'embeddings_model': EmbeddingsModelUniversalSentenceEncoder(model_name)}


def text_embeddings_doc2vec(input_dict):
    languages = {
        'en': 'doc2vec.bin',
    }
    model_name = text_embeddings_extract_model_name(input_dict, languages)
    return {'embeddings_model': EmbeddingsModelDoc2Vec(model_name)}


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
    adc = input_dict['adc']
    token_annotation = input_dict['token_annotation'] or 'TextBlock'
    documents = adc.documents
    bow_dataset = embeddings_model.apply(documents, token_annotation)
    return {'bow_dataset': bow_dataset}
