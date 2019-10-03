from os import makedirs, path
from pathlib import Path

import numpy as np
from Orange.data import ContinuousVariable, DiscreteVariable, Domain

PROJECT_DATA_DIR = path.join(str(Path.home()), '.cf_text_embeddings')


def orange_domain(n_features, unique_labels):
    return Domain([ContinuousVariable.make('Feature %d' % i) for i in range(n_features)],
                  DiscreteVariable('class', values=unique_labels))


def load_numpy_array(path_):
    return np.load(path_)


def extract_map_invert_y(odt):
    unique_labels = odt.domain.class_var.values
    mapping = {v: k for v, k in enumerate(unique_labels)}
    y = np.array([mapping[int(label)] for label in odt.Y])
    return y, unique_labels


def map_y(y):
    unique_labels = list(sorted(set(y)))
    mapping = {v: k for k, v in enumerate(unique_labels)}
    y = np.array([mapping[label] for label in y])
    return y, unique_labels


def to_int(num_string, default):
    try:
        num = int(num_string)
    except ValueError:
        return default
    return num


def to_float(num_string, default):
    try:
        num = float(num_string)
    except ValueError:
        return default
    return num


def ensure_dir(f):
    d = path.dirname(f)
    if not path.exists(d):
        makedirs(d, exist_ok=True)


def get_media_root():
    from mothra.settings import MEDIA_ROOT
    return MEDIA_ROOT


def save_numpy_array(dirname, filename, array):
    filepath = path.join(dirname, filename)
    ensure_dir(filepath)
    np.save(filepath, array)
