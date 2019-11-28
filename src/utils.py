import pandas as pd
import numpy as np
from pathlib import Path
import pickle


class Logger:

    def __init__(self, quiet=False):
        self.quiet = quiet

    def i(self, msg):
        if not self.quiet:
            print(msg)

    def d(self, msg):
        if not self.quiet:
            print(msg)

    def e(self, msg):
        if not self.quiet:
            print(msg)


logger = Logger()


def smooth(y, half_window):
    smoothed = [0] * len(y)
    for i in range(len(y)):
        start = max(0, i - half_window)
        end = min(len(y) - 1, i + half_window)
        smoothed[i] = np.mean(y[start:end])
    return smoothed


def cache(func, args=(), force=False, path=Path('./tmp.pickle')):
    """ Cache function result """
    path = Path(path)

    if path.exists() and path.is_file() and not force:
        try:
            result = pickle.load(path.open('rb'))
        except IOError:
            print('Could not load pickle %s' % path)
            result = None

    else:
        result = func(*args)
        try:
            pickle.dump(result, path.open('wb'))
        except IOError:
            print('Could not save pickle %s' % path)

    return result


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
