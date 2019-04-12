from text_embeddings.embeddings_model import EmbeddingsModel, ModelType


def text_embeddings_common(input_dict, model_type, languages, **kwargs):
    lang = input_dict['lang']
    model_name = languages.get(lang)
    if model_name is None:
        raise Exception('%s model for %s language is not supported' % (model_type, lang))
    return {'embeddings_model': EmbeddingsModel(model_type, model_name, **kwargs)}


def text_embeddings_word2vec(input_dict):
    model_type = ModelType.word2vec
    languages = {
        'en': 'GoogleNews-vectors-negative300.wv.bin',
        'es': 'SBW-vectors-300-min5.wv.bin',
    }
    return text_embeddings_common(input_dict, model_type, languages)


def text_embeddings_glove(input_dict):
    model_type = ModelType.glove
    languages = {
        'en': 'glove.6B.300d.wv.bin',
    }
    return text_embeddings_common(input_dict, model_type, languages)


def text_embeddings_fasttext(input_dict):
    model_type = ModelType.fasttext
    languages = {
        # model downloaded and converted with gensim:
        # https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
        'en': 'wiki-news-300d-1M.bin',
    }
    return text_embeddings_common(input_dict, model_type, languages)


def text_embeddings_universal_sentence_encoder(input_dict):
    model_type = ModelType.universal_sentence_encoder
    languages = {
        # https://tfhub.dev/google/universal-sentence-encoder/2
        'en': 'english',
    }
    return text_embeddings_common(input_dict, model_type, languages)


def text_embeddings_doc2vec(input_dict):
    model_type = ModelType.doc2vec
    languages = {
        'en': 'doc2vec.bin',
    }
    return text_embeddings_common(input_dict, model_type, languages)


def text_embeddings_elmo(input_dict):
    model_type = ModelType.elmo
    languages = {
        # https://tfhub.dev/google/elmo/2
        'en': 'english',
    }
    return text_embeddings_common(input_dict, model_type, languages, model_output='elmo',
                                  signature="default", as_dict=True)


def text_embeddings_embeddings_hub(input_dict):
    embeddings_model = input_dict['embeddings_model']
    adc = input_dict['adc']
    token_annotation = input_dict['token_annotation'] or 'TextBlock'
    documents = adc.documents

    bow_dataset = embeddings_model.apply(documents, token_annotation)
    return {'bow_dataset': bow_dataset}
