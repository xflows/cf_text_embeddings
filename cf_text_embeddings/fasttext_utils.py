import os
import tempfile
import zlib

import numpy as np
import fasttext
import editdistance


def load_packed_model(cdata):
    with tempfile.TemporaryDirectory() as tmpdirname:
        modelpath = os.path.join(tmpdirname, 'model.bin')
        with open(modelpath, 'wb') as fp:
            fp.write(zlib.decompress(cdata))
        model = fasttext.load_model(modelpath)
    return model


def check_editdistance(target_word_list, compare_word, treshold=0.2):
    farEnough = True
    for w in target_word_list:
        ed = 1 - (float(editdistance.eval(w, compare_word)) / max(len(w), len(compare_word)))
        if ed > treshold:
            farEnough = False
            break
    return farEnough


def get_full_matrix(model, normalize_rows=True):
    vectors = [model.get_word_vector(word) for word in model.get_words()]
    if normalize_rows:
        vectors = [vec/np.linalg.norm(vec) for vec in vectors]
    return np.array(vectors)


def get_nearest_neighbors_from_vector(model, vector, matrix=None, k=10):
    if matrix is None:
        matrix = get_full_matrix(model, normalize_rows=True)  # matrix of all vectors
    vector = vector/np.linalg.norm(vector)  # normalize
    cosims = np.dot(matrix, vector)  # cosine sim to all
    imaxs = np.argsort(-cosims)[:k]  # indices of maximums
    words = model.get_words()
    return [(cosims[i], words[i]) for i in imaxs]


def format_word_expression(s, modelVarName='model'):
    new = []
    words = []
    for elt in s.split():
        if elt in '+-':
            new.append(elt)
        else:
            words.append(elt)
            new.append('{}.get_word_vector("{}")'.format(modelVarName, elt))
    return words, ' '.join(new)
