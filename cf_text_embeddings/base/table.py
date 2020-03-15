from Orange.data import ContinuousVariable, DiscreteVariable, Domain, Table


def orange_domain(n_features, unique_labels):
    return Domain([ContinuousVariable.make('Feature %d' % i) for i in range(n_features)],
                  DiscreteVariable('class', values=unique_labels))


def orange_data_table(embeddings, labels):
    uniq_labels = list(sorted(set(labels)))
    y = [uniq_labels.index(r) for r in labels]  # binarize labels
    domain = orange_domain(embeddings.shape[1], uniq_labels)
    table = Table(domain, embeddings, Y=y)
    return table
