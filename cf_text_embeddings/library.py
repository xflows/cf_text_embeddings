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
    from sklearn.utils import Bunch
    dataset = Bunch(data=input_dict['X'], target=input_dict['y'],
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
    counts = Counter([langdetect.detect(doctext[:100]) for doctext in input_dict['corpus']]).most_common()
    return {'main_lang': counts[0][0], 'other_langs': [x[0] for x in counts[1:]]}


def cf_text_embeddings_train_fasttext(input_dict):
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
                                            ws=int(input_dict['window']),
                                            lr=float(input_dict['lrate']),
                                            bucket=int(input_dict['bucket']),
                                            epoch=int(input_dict['epoch'])
                                            )
        modelpath = os.path.join(tmpdirname, 'model.bin')
        model.save_model(modelpath)
        data = open(modelpath, 'rb').read()
        cdata = zlib.compress(data, level=1)
    return {'model': cdata}


def cf_text_embeddings_neighbouring_words(input_dict):
    from . import fasttext_utils as futils

    model = futils.load_packed_model(input_dict['model'])

    k = int(input_dict['k'])
    threshold = float(input_dict['threshold'])
    result = []
    for word in input_dict['words']:
        if len(word.split()) != 1:
            raise ValueError('Invalid input: more than one word per line!')

        word_neighbours = [word]
        for d, candidate in model.get_nearest_neighbors(word, k*50):  # rule of thumb
            if futils.check_editdistance(word_neighbours, candidate, threshold):
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


def cf_text_embeddings_evaluate_word_expressions(input_dict):
    import copy
    from . import fasttext_utils as futils

    corpus = input_dict['corpus']
    model = futils.load_packed_model(input_dict['model'])
    threshold = float(input_dict['threshold'])
    k = int(input_dict['k'])

    fullMatrix = futils.get_full_matrix(model, normalize_rows=True)
    output = []
    for exp in corpus:
        words, expression = futils.format_word_expression(exp, modelVarName='model')
        nwords = len(words)
        result_vector = eval(expression)
        exp_results = copy.copy(words)
        for d, candidate in futils.get_nearest_neighbors_from_vector(model, result_vector, matrix=fullMatrix, k=k*50):  # rule of thumb...
            if futils.check_editdistance(exp_results, candidate, threshold):
                exp_results.append(candidate)
                if len(exp_results) == k + nwords:
                    break
        output.append(exp_results[nwords:])
    return {'results': output}


def cf_text_embeddings_apply_trained_fasttext(input_dict):
    from . import fasttext_utils as futils

    model = futils.load_packed_model(input_dict['model'])
    corpus = input_dict['corpus']
    agg = input_dict['aggregation']

    result = []
    for doc in corpus:
        tokvecs = np.array([model.get_word_vector(token) for token in doc])
        if agg == 'sum':
            docvec = tokvecs.sum(axis=0)
        elif agg == 'average':
            docvec = tokvecs.mean(axis=0)
        result.append(docvec)
    return {'embedding': np.array(result)}


def cf_text_embeddings_read_zip_corpus(input_dict):
    import zipfile
    fname = input_dict['archive']
    ftypes = input_dict['ftypes'].replace(',', ' ').split()

    corpus = []
    with zipfile.ZipFile(fname) as archive:
        for fname in archive.namelist():
            if any([fname.endswith(x) for x in ftypes]):
                with archive.open(fname) as fp:
                    corpus.append(fp.read().decode('utf-8').strip())
    return {'corpus': corpus}
