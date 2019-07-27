# removing features with only one unique value

def feats_n_unique(data, n_unique):
    unique_counts = data.nunique(axis=0)
    feats_with_unique = unique_counts[unique_counts < n_unique].index
    return feats_with_unique


def remove_new_values(train, test):
    '''

    :param train: The train set dataframe
    :param test: The test set dataframe without the TARGET variable
    :return: The test set with new unseen values pruned

    This function would remove any unseen values in test that were
    not seen in train for all the columns.

    '''
    train_describe = train.describe()
    test_describe = test.describe()
    for c in test.columns:
        min_cut = train_describe[c]['min']
        max_cut = train_describe[c]['max']

        if test_describe[c]['min'] < min_cut:
            test.loc[test[c] < min_cut, c] = min_cut

        if test_describe[c]['max'] > max_cut:
            test.loc[test[c] > max_cut, c] = max_cut


    return test