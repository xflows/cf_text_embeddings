import os
import tempfile
import zlib
import numpy as np
import pandas as pd
import fasttext
# from gensim.test.utils import datapath
# from gensim.models.fasttext import load_facebook_model

from lemmagen3 import Lemmatizer
import langdetect
import editdistance

from .base import io, tokenizers
from .base.common import load_numpy_array, map_checkbox_value, to_float, to_int
from .base.embeddings_model import (cf_text_embeddings_model_path,
                                    TFIDF, EmbeddingsModelBert,
                                    EmbeddingsModelBertEmbeddia,
                                    EmbeddingsModelDoc2Vec,
                                    EmbeddingsModelElmo,
                                    EmbeddingsModelElmoAllen,
                                    EmbeddingsModelFastText,
                                    EmbeddingsModelFastTextEmbedia,
                                    EmbeddingsModelFastTextHr,
                                    EmbeddingsModelFastTextSl,
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
    assert 'to_lowercase' in input_dict, 'To lowercase argument is missing'

    to_lowercase = map_checkbox_value(input_dict['to_lowercase'])
    texts = tokenizers.toktok_tokenizer(input_dict['texts'], to_lowercase=to_lowercase)
    return {'texts': texts}


def cf_text_embeddings_regex_word_tokenizer(input_dict):
    assert 'texts' in input_dict, 'Text is missing'
    assert 'to_lowercase' in input_dict, 'To lowercase argument is missing'

    to_lowercase = map_checkbox_value(input_dict['to_lowercase'])
    texts = tokenizers.regex_word_tokenizer(input_dict['texts'], to_lowercase=to_lowercase)
    return {'texts': texts}


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
    assert 'filter_extremes' in input_dict, 'filter_extremes is missing'
    assert 'no_below' in input_dict, 'no_below is missing'
    assert 'no_above' in input_dict, 'no_above is missing'

    filter_extremes = map_checkbox_value(input_dict['filter_extremes'])
    no_below = to_int(input_dict['no_below'], 5)
    no_above = to_float(input_dict['no_above'], 0.5)

    tfidf = TFIDF()
    tfidf.train(input_dict['texts'], filter_extremes, no_below, no_above)
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


def cf_text_embeddings_fasttext_sl(input_dict):
    input_dict['lang'] = 'sl'
    return cf_text_embeddings_base(EmbeddingsModelFastTextSl, input_dict)


def cf_text_embeddings_fasttext_embeddia(input_dict):
    return cf_text_embeddings_base(EmbeddingsModelFastTextEmbedia, input_dict)


def cf_text_embeddings_fasttext_hr(input_dict):
    input_dict['lang'] = 'hr'
    return cf_text_embeddings_base(EmbeddingsModelFastTextHr, input_dict)


def cf_text_embeddings_elmo(input_dict):
    return cf_text_embeddings_base(EmbeddingsModelElmo, input_dict)


def cf_text_embeddings_elmo_embedia(input_dict):
    return cf_text_embeddings_base(EmbeddingsModelElmoAllen, input_dict)


def cf_text_embeddings_lsi(input_dict):
    texts = input_dict['texts']
    num_topics_default = 200
    num_topics = to_int(input_dict['num_topics'], num_topics_default)
    num_topics = num_topics_default if num_topics < 1 else num_topics
    decay = to_float(input_dict['decay'], 1.0)
    train_on_tfidf = map_checkbox_value(input_dict['train_on_tfidf'])
    filter_extremes = map_checkbox_value(input_dict['filter_extremes'])

    embeddings_model = EmbeddingsModelLSI(num_topics, decay, train_on_tfidf,
                                          filter_extremes=filter_extremes)
    embeddings = embeddings_model.apply(texts)
    return {'embeddings': embeddings}


def cf_text_embeddings_bert(input_dict):
    texts = input_dict['texts']

    model = input_dict['model_selection']
    embeddings_model = EmbeddingsModelBert(model)
    embeddings = embeddings_model.apply(texts)
    return {'embeddings': embeddings}


def cf_text_embeddings_bert_embeddia(input_dict):
    texts = input_dict['texts']

    model = input_dict['model_selection']
    embeddings_model = EmbeddingsModelBertEmbeddia(model)
    embeddings = embeddings_model.apply(texts)
    return {'embeddings': embeddings}


def cf_text_embeddings_bert_hate_speech(input_dict):
    import torch
    import transformers as tr

    tokenizer = tr.BertTokenizer.from_pretrained('bert-base-uncased')
    model_path = cf_text_embeddings_model_path('multi', 'ml_bert_hate_speech')
    model = tr.BertForSequenceClassification.from_pretrained(model_path)

    idx2label = {0: 'hate', 1: 'normal'}
    labels = []
    probabilities = []
    for doc in input_dict['texts']:
        tokenized = tokenizer.encode_plus(doc, return_tensors='pt', add_special_tokens=True, max_length=256, truncation=True)
        result = model(input_ids=tokenized['input_ids'], attention_mask=tokenized['attention_mask'], return_dict=True)
        probs = torch.nn.functional.softmax(result['logits'], dim=1)[0].tolist()
        probabilities.append(probs)
        labels.append(idx2label[np.argmax(probs)])
    return {'labels': labels, 'probabilities': probabilities}


def cf_text_embeddings_bert_sentiment(input_dict):
    import torch
    import transformers as tr

    # This is how Andraz defined labels... (negative 0, neutral 1, positive 2)
    idx2label = {i: x for i, x in enumerate(sorted(['positive', 'negative', 'neutral']))}
    tokenizer = tr.BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model_path = cf_text_embeddings_model_path('multi', 'ml_bert_sentiment')
    config = tr.BertConfig.from_pretrained(os.path.join(model_path, 'config.json'), num_labels=len(idx2label))
    model = tr.BertForSequenceClassification.from_pretrained(model_path, config=config)

    labels = []
    probabilities = []
    for doc in input_dict['texts']:
        tokenized = tokenizer.encode_plus(doc, return_tensors='pt', add_special_tokens=True, max_length=512, truncation=True)
        result = model(input_ids=tokenized['input_ids'], attention_mask=tokenized['attention_mask'], return_dict=True)
        probs = torch.nn.functional.softmax(result['logits'], dim=1)[0].tolist()
        probabilities.append(probs)
        labels.append(idx2label[np.argmax(probs)])
    return {'labels': labels, 'probabilities': probabilities}


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


def cf_text_embeddings_concatenate_embeddings(input_dict):
    arrays = input_dict['embedding']
    return {'concat_embeddings': np.concatenate(arrays, axis=1)}


def cf_text_embeddings_make_scikit_bunch(input_dict):
    from sklearn.datasets import base as ds
    dataset = ds.Bunch(data=input_dict['X'], target=input_dict['y'],
                       feature_names=input_dict['feature_names'], DESCR=input_dict['description'])
    return {'dataset': dataset}


def cf_text_embeddings_detokenize(input_dict):
    return {'doctexts': [' '.join(doctokens) for doctokens in input_dict['tokenized_docs']]}


def cf_text_embeddings_lemmatize_lemmagen(input_dict):
    supported = ['bg', 'cs', 'de', 'en', 'es', 'et', 'fa', 'fr', 'hr', 'hu', 'it', 'mk', 'pl', 'ro', 'ru', 'sk', 'sl', 'sr', 'uk']
    lang = input_dict['language']
    if lang not in supported:
        raise('Language {} is not supported in Lemmagen'.format(lang))
    lem = Lemmatizer(input_dict['language'])
    return{'lemmatized_docs': [[lem.lemmatize(token) for token in doctokens] for doctokens in input_dict['tokenized_docs']]}


def cf_text_embeddings_detect_language(input_dict):
    from collections import Counter
    counts = Counter([langdetect.detect(doctext) for doctext in input_dict['corpus']]).most_common()
    return {'main_lang': counts[0][0], 'other_langs': [x[0] for x in counts[1:]]}


def cf_text_embeddings_train_fasttext(input_dict):
    # from collections import Counter

    corpus = input_dict['corpus']
    if isinstance(corpus[0], str):
        text = '\n'.join(corpus)
    elif isinstance(corpus[0], list):
        text = '\n'.join([' '.join(doctokens) for doctokens in corpus])
    else:
        raise ValueError('Invalid corpus data.')

    with tempfile.TemporaryDirectory() as tmpdirname:
        corpuspath = os.path.join(tmpdirname, 'corpus.txt')
        with open(corpuspath, 'w') as fp:
            fp.write(text)
        model = fasttext.train_unsupervised(corpuspath,
                                            model=input_dict['model'],
                                            dim=int(input_dict['dimension']),
                                            minCount=int(input_dict['minCount']),
                                            ws=int(input_dict['window']))
        modelpath = os.path.join(tmpdirname, 'model.bin')
        model.save_model(modelpath)
        data = open(modelpath, 'rb').read()
        cdata = zlib.compress(data, level=1)
    return {'model': cdata}

    # if input_dict['lemmatize']:
    #     counts = Counter([langdetect.detect(doctext) for doctext in input_dict['corpus']]).most_common()
    #     lang = counts[0][0]
    #     lemmatizer = Lemmatizer(lang)
    #     lem_corpus = ['\n'.join([' '.join([lemmatizer.lemmatize(token) for token in block.split()]) for block in doctext.split('\n')]) for doctext in input_dict['corpus']]
    #     corpus = lem_corpus
    # else:
    #     corpus = input_dict['corpus']
    #
    # with tempfile.TemporaryDirectory() as tmpdirname:
    #     corpuspath = os.path.join(tmpdirname, 'corpus.txt')
    #     with open(corpuspath, 'w') as fp:
    #         fp.write('\n'.join(corpus))
    #     model = fasttext.train_unsupervised(corpuspath,
    #                                         model=input_dict['model'],
    #                                         dim=int(input_dict['dimension']),
    #                                         minCount=int(input_dict['minCount']),
    #                                         ws=int(input_dict['window']))
    #     modelpath = os.path.join(tmpdirname, 'model')
    #     model.save_model(modelpath)
    #     fb_model = load_facebook_model(modelpath)
    # return {'model': fb_model}


def cf_text_embeddings_neighbouring_words(input_dict):

    def filter_too_similar_by_editdistance(wordlist, compare_word, threshold=0.2):
        return [w for w in wordlist if (1 - (float(editdistance.eval(w, compare_word)) / max(len(w), len(compare_word)))) <= threshold]

    def check_editdistance(target_word_list, compare_word, treshold=0.2):
        farEnough = True
        for w in target_word_list:
            ed = 1 - (float(editdistance.eval(w, compare_word)) / max(len(w), len(compare_word)))
            if ed > treshold:
                farEnough = False
                break
        return farEnough


    cdata = input_dict['model']
    with tempfile.TemporaryDirectory() as tmpdirname:
        modelpath = os.path.join(tmpdirname, 'model.bin')
        with open(modelpath, 'wb') as fp:
            fp.write(zlib.decompress(cdata))
        model = fasttext.load_model(modelpath)

    k = int(input_dict['k'])
    threshold = float(input_dict['threshold'])
    result = []
    for word in input_dict['words']:
        if len(word.split()) != 1:
            raise ValueError('Invalid input: more than one word per line!')

        word_neighbours = [word]
        for d, candidate in model.get_nearest_neighbors(word, k*50):  # rule of thumb
            if check_editdistance(word_neighbours, candidate, threshold):
                word_neighbours.append(candidate)
                if len(word_neighbours) == k + 1:
                    break
        result.append(word_neighbours)
    return {'neighbours': result}


def cf_text_embeddings_token_frequency(input_dict):
    from collections import Counter
    corpus = input_dict['corpus']
    counts = Counter()
    for doctokens in corpus:
        counts.update(doctokens)
    return {'freqs': [list(x) for x in counts.most_common()]}


def cf_text_embeddings_corpus2dataframe(input_dict):
    return {'df': pd.DataFrame().assign(**{'document': input_dict['corpus']})}


def cf_text_embeddings_append_column(input_dict):
    df = input_dict['df']
    name = input_dict['column_name']
    if not name:
        raise ValueError('New column name not set!')
    data = input_dict['column_data']
    return {'df': df.assign(**{name: pd.Series(data)})}


def cf_text_embeddings_dataframe2csv(input_dict):
    return input_dict
