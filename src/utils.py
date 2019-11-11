import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
from tqdm import tqdm


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


def preprocess_all(data, config):
    data = np.array(data[config['features']])

    trn_samples = round(config['train_test_split'] * data.shape[0])
    data_train = data[:trn_samples, :]
    data_test = data[trn_samples:, :]

    seqs_train = create_dataset_all(data_train, config)
    seqs_test = create_dataset_all(data_test, config)

    return seqs_train, seqs_test


def preprocess_all_extra(data, config):
    data = np.array(data[config['features']])
    # Add ARIMA feature to the dataset
    #data = np.concatenate([data, extract_ARIMA(data)])

    trn_samples = round(config['train_test_split'] * data.shape[0])
    data_train = data[:trn_samples, :]
    data_test = data[trn_samples:, :]

    seqs_train = create_dataset_all_extra(data_train, config)
    seqs_test = create_dataset_all_extra(data_test, config)

    return seqs_train, seqs_test


def create_dataset(matrix, config):
    dat_x = []
    dat_y = []
    for i in range(matrix.shape[0]-config['length_sequence']):  # +1):
        aux_matrix = matrix[i:i+config['length_sequence'], :]
        aux_matrix = aux_matrix / aux_matrix[0, :] - 1

        dat_x.append(aux_matrix[:-1, :])
        index = np.argwhere(np.array(config['features']) == config['feature_prediction'])[0][0]
        dat_y.append(aux_matrix[-1, index])

    return list(zip(dat_x, dat_y))


def create_dataset_all(matrix, config):
    dat_x = []
    dat_y = []
    for i in range(matrix.shape[0]-config['length_sequence']):  # +1):
        aux_matrix = matrix[i:i+config['length_sequence'], :]
        aux_matrix = aux_matrix / aux_matrix[0, :] - 1

        dat_x.append(aux_matrix[:-1, :])
        dat_y.append(aux_matrix[-1, :])

    return list(zip(dat_x, dat_y))


def create_dataset_all_extra(matrix, config):
    dat_x = []
    dat_y = []
    coefs = config['fourier_coefs']
    for i in range(matrix.shape[0]-config['length_sequence']):  # +1):
        aux_matrix = matrix[i:i+config['length_sequence'], :]

        fft = np.fft.fft(aux_matrix[:, config['features'].index('Close')])

        for coef in coefs:
            aux_fft = np.copy(fft)
            aux_fft[coef:-coef] = 0
            ifft = np.fft.ifft(aux_fft).real.reshape((len(aux_fft), 1))
            aux_matrix = np.concatenate([aux_matrix, ifft], axis=1)

        #fig = plt.figure()
        #fig.suptitle('Fourier Feats')
        #sp = fig.add_subplot(1, 1, 1)
        #sp.plot(list(range(len(ifft))), ifft, linewidth=0.5)
        #sp.plot(list(range(len(ifft))), aux_matrix[:, config['features'].index('Close')], 'r', linewidth=0.5)

        aux_matrix = aux_matrix / aux_matrix[0, :] - 1

        dat_x.append(aux_matrix[:-1, :])
        dat_y.append(aux_matrix[-1, :])

    return list(zip(dat_x, dat_y))


def create_dataset_lstm_global(lstm_out, arima_out, gt):
    dat_x = []
    dat_y = []
    for i in range(len(lstm_out)):
        dat_x.append(np.concatenate((lstm_out[i].reshape(len(lstm_out[i]), 1), arima_out[i]), axis=1))
        dat_y.append(gt[i * len(lstm_out[i]):(i+1) * len(lstm_out[i])])

    return list(zip(dat_x, dat_y))


def create_dataset_lstm_global2(lstm_out, arima_out):
    dat_x = []
    for i in range(len(lstm_out)):
        dat_x.append(np.array([lstm_out[i][0], arima_out[i][0][0]]))

    return np.array(dat_x)


def create_dataset_reg(lstm_out, arima_out, gt):
    dat_x = []
    dat_y = []
    for i in range(len(lstm_out)):
        for j in range(len(lstm_out[i])):
            dat_x.append(np.array([lstm_out[i][j], arima_out[i][j][0]]))
            dat_y.append(gt[i+j])

    return np.array(dat_x), dat_y


def create_gen(seqs_train, batch_size):
    i = 0

    while i < len(seqs_train):
        if len(seqs_train) - i <= batch_size:
            trn_x, trn_y = zip(*seqs_train[i:])
            yield np.array(trn_x), np.array(trn_y)

            i = 0

        trn_x, trn_y = zip(*seqs_train[i:i+batch_size])

        yield np.array(trn_x), np.array(trn_y)

        i += batch_size


def color_gen():
    colors = list('rgmykc')
    while True:
        for c in colors:
            yield c


def predict_arima(tst_data, config, next_k_items=50, plot=None):
    y_preds_arima = []
    y_pred_arima = np.zeros((next_k_items, 1))
    color = color_gen()
    counter = 0

    for seq_id in tqdm(range(len(tst_data))):
        if counter % next_k_items == 0:
            counter = 0
            history = np.copy(tst_data[seq_id][0][:, config['features'].index('Close')])

        else:
            history = np.concatenate((tst_data[seq_id][0][:-counter, config['features'].index('Close')],
                                      y_pred_arima[:counter].reshape([counter,])))

        arima = ARIMA(history, order=tuple(config['ARIMA_params']))
        arima_fit = arima.fit(disp=0)

        y_pred_arima[counter] = arima_fit.forecast()[0]

        counter += 1

        if counter == next_k_items:
            y_preds_arima.append(np.copy(y_pred_arima))

            if plot is not None:
                end = seq_id + config['length_sequence'] - next_k_items
                start = end - next_k_items
                plot.plot(list(range(start, end)), y_pred_arima[:, 0], next(color) + '*-', markersize=2, linewidth=0.5)

    return y_preds_arima


def predict_arima_all(tst_data, config, next_k_items=50, plot=None):
    y_preds_arima = []
    y_pred_arima = np.zeros((next_k_items, 1))

    for seq_id in tqdm(range(len(tst_data))):
        for counter in range(next_k_items):
            if counter % next_k_items == 0:
                counter = 0
                history = np.copy(tst_data[seq_id][0][:, config['features'].index('Close')])

            else:
                history = np.concatenate((tst_data[seq_id][0][:-counter, config['features'].index('Close')],
                                          y_pred_arima[:counter].reshape([counter,])))

            arima = ARIMA(history, order=tuple(config['ARIMA_params']))
            arima_fit = arima.fit(disp=0)

            y_pred_arima[counter] = arima_fit.forecast()[0]

        y_preds_arima.append(np.copy(y_pred_arima))

    return y_preds_arima
