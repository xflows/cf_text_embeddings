import numpy as np
from Orange.data import ContinuousVariable, DiscreteVariable, Domain


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
