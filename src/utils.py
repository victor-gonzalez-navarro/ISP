import pandas as pd
import numpy as np


def read_data(path):
    data = pd.read_csv(path, delimiter=',')

    return data


def preprocess(data, config):
    data = np.array(data[config['features']])

    trn_samples = round(config['train_test_split'] * data.shape[0])
    data_train = data[:trn_samples, :]
    data_test = data[trn_samples:, :]

    seqs_train = create_dataset(data_train, config)
    seqs_test = create_dataset(data_test, config)

    return seqs_train, seqs_test


def create_dataset(matrix, config):
    dat_x = []
    dat_y = []
    for i in range(matrix.shape[0]-config['length_sequence']+1):
        aux_matrix = matrix[i:i+config['length_sequence'], :]
        aux_matrix = aux_matrix / aux_matrix[0, :] - 1

        dat_x.append(aux_matrix[:-1, :])
        index = np.argwhere(np.array(config['features']) == config['feature_prediction'])[0][0]
        dat_y.append(aux_matrix[-1, index])

    return list(zip(dat_x, dat_y))


def create_gen(seqs_train, batch_size):
    i = 0

    while i < len(seqs_train):
        if len(seqs_train) - i <= batch_size:
            trn_x, trn_y = zip(*seqs_train[i:])
            yield np.array(trn_x), np.array(trn_y)

            i = 0

        trn_x, trn_y = zip(*seqs_train[i:i+batch_size])

        yield np.array(trn_x), np.array(trn_y)

        i += 1


def color_gen():
    colors = list('rgmykc')
    while True:
        for c in colors:
            yield c
