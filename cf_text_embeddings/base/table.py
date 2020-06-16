from Orange.data import ContinuousVariable, DiscreteVariable, Domain, Table


# def orange_domain(n_features, unique_labels):
#     return Domain([ContinuousVariable.make('Feature %d' % i) for i in range(n_features)],
#                   DiscreteVariable('class', values=unique_labels))


def orange_data_table(embeddings, labels):
    if labels == [] or labels is None:
        domain = Domain([ContinuousVariable.make('Feature %d' % i) for i in range(embeddings.shape[1])])
        y = None
    else:
        uniq_labels = list(sorted(set(labels)))
        y = [uniq_labels.index(r) for r in labels]
        domain = Domain([ContinuousVariable.make('Feature %d' % i) for i in range(embeddings.shape[1])],
                        DiscreteVariable('class', values=uniq_labels))
    table = Table(domain, embeddings, Y=y)
    return table
