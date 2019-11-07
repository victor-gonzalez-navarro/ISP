import json
import warnings
import pickle
import matplotlib.pyplot as plt

from lstm_global import LSTM_Global
from model2 import Model2
from utils import preprocess, preprocess_all, preprocess_all_extra, color_gen, predict_arima, create_dataset_lstm_global
from utils import read_data
from utils import create_gen
from model import Model
import datetime
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from tqdm import tqdm


def main_new_model():
    with open('../config/params.json', 'r') as read_file:
        config = json.load(read_file)

    data = read_data('../data/sp500.csv')
    seqs_train, seqs_test = preprocess(data, config)

    gen = create_gen(seqs_train, config['batch_size'])

    fig = plt.figure()
    fig.suptitle('Model 1')
    sp = fig.add_subplot(1, 1, 1)
    sp.plot(list(range(len(seqs_test))), [y for x, y in seqs_test], linewidth=0.5)

    m = Model(config, name='Model_{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now()))
    m.buildLayers()
    m.fit(gen, len(seqs_train))

    m.predict(seqs_test, next_k_items=config['next_k_items'], plot=sp)

    plt.show()


def main_new_model_all():
    with open('../config/params2.json', 'r') as read_file:
        config = json.load(read_file)

    data = read_data('../data/sp500.csv')
    seqs_train, seqs_test = preprocess_all(data, config)

    gen = create_gen(seqs_train, config['batch_size'])

    fig = plt.figure()
    fig.suptitle('Model 2')
    sp = fig.add_subplot(1, 1, 1)
    sp.plot(list(range(len(seqs_test))), [y[0] for x, y in seqs_test], linewidth=0.5)

    m = Model2(config, name='Model2_{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now()))
    m.buildLayers()
    m.fit(gen, len(seqs_train))

    m.predict(seqs_test, next_k_items=config['next_k_items'], plot=sp)

    plt.show()


def main_best_model():
    with open('../config/params.json', 'r') as read_file:
        config = json.load(read_file)

    data = read_data('../data/sp500.csv')
    seqs_train, seqs_test = preprocess(data, config)

    fig = plt.figure()
    fig.suptitle('Model 1')
    sp = fig.add_subplot(1, 1, 1)
    sp.plot(list(range(len(seqs_test))), [y for x, y in seqs_test], linewidth=0.5)

    m = Model(config, model_path='../models/11102019-120649-e2.h5')
    m.predict(seqs_test, next_k_items=config['next_k_items'], plot=sp)

    plt.show()

    print('Finished')


def main_best_model_all():
    with open('../config/params2.json', 'r') as read_file:
        config = json.load(read_file)

    data = read_data('../data/sp500.csv')
    seqs_train, seqs_test = preprocess_all(data, config)

    fig = plt.figure()
    fig.suptitle('Model 2')
    sp = fig.add_subplot(1, 1, 1)
    sp.plot(list(range(len(seqs_test))), [y[0] for x, y in seqs_test], linewidth=0.5)

    m = Model2(config, model_path='../models/Model2_2019-10-17_14_35_54')
    m.predict(seqs_test, next_k_items=config['next_k_items'], plot=sp)

    plt.show()

    print('Finished')


def main_fourier():
    with open('../config/params2.json', 'r') as read_file:
        config = json.load(read_file)

    data = read_data('../data/sp500.csv')

    seqs_train, seqs_test = preprocess_all_extra(data, config)

    gen = create_gen(seqs_train, config['batch_size'])

    fig = plt.figure()
    fig.suptitle('Model 2')
    sp = fig.add_subplot(1, 1, 1)
    sp.plot(list(range(len(seqs_test))), [y[0] for x, y in seqs_test], linewidth=0.5)

    m = Model2(config, name='Model2_{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now()))
    output_feats = seqs_train[0][0].shape[1]
    m.buildLayers(output_feats)
    m.fit(gen, len(seqs_train))

    m.predict(seqs_test, next_k_items=config['next_k_items'], plot=sp)

    plt.show()

    print('Finished')


def main_best_fourier():
    pass


def main_ARIMA():
    warnings.filterwarnings("ignore")

    with open('../config/params2.json', 'r') as read_file:
        config = json.load(read_file)

    data = read_data('../data/sp500.csv')

    seqs_train, seqs_test = preprocess_all_extra(data, config)

    gen = create_gen(seqs_train, config['batch_size'])

    fig = plt.figure()
    fig.suptitle('Model 2')
    sp = fig.add_subplot(1, 1, 1)
    sp.plot(list(range(len(seqs_test))), [y[0] for x, y in seqs_test], linewidth=2)

    m = Model2(config, name='Model2_{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now()))
    output_feats = seqs_train[0][0].shape[1]
    m.buildLayers(output_feats)
    print('Fitting LSTM model...')
    m.fit(gen, len(seqs_train))
    print('Obtaining Train Predictions by LSTM model...')
    lstm_preds_trn = m.predict(seqs_train, next_k_items=config['next_k_items'])

    print('Fitting ARIMA model...')
    arima_preds_trn = predict_arima(seqs_train, config, next_k_items=config['next_k_items'])

    f = open('./arima_preds.pckl', 'wb')
    pickle.dump(arima_preds_trn, f)
    f.close()

    f2 = open('./lstm_preds.pckl', 'wb')
    pickle.dump(lstm_preds_trn, f2)
    f2.close()

def main_other():
    warnings.filterwarnings("ignore")

    with open('../config/params2.json', 'r') as read_file:
        config = json.load(read_file)

    data = read_data('../data/sp500.csv')

    seqs_train, seqs_test = preprocess_all_extra(data, config)

    m = Model2(config, model_path='../models/Model2_2019-11-07_16_31_01')

    f = open('./arima_preds.pckl', 'rb')
    arima_preds_trn = pickle.load(f)
    f.close()

    f2 = open('./lstm_preds.pckl', 'rb')
    lstm_preds_trn = pickle.load(f2)
    f2.close()

    fig = plt.figure()
    fig.suptitle('Model 2')
    sp = fig.add_subplot(1, 1, 1)
    sp.plot(list(range(len(seqs_test))), [y[0] for x, y in seqs_test], linewidth=2)

    trn_samples = round(config['train_test_split'] * data.shape[0])
    seqs_train2 = create_dataset_lstm_global(lstm_preds_trn,
                                             arima_preds_trn,
                                             data[config['features']].values[:(trn_samples + config['next_k_items']),
                                                  config['features'].index('Close')])

    gen2 = create_gen(seqs_train2, config['batch_size'])

    lstm_global = LSTM_Global(config, name='LSTM_Global_{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now()))
    lstm_global.buildLayers()
    print('Fitting LSTM Global model...')
    lstm_global.fit(gen2, len(seqs_train2))

    print('Starting ARIMA Predictions...')
    arima_preds_tst = predict_arima(seqs_test, config, next_k_items=config['next_k_items'], plot=sp)
    print('ARIMA Predictions Finished!')

    print('Starting LSTM Predictions...')
    lstm_preds_tst = m.predict(seqs_test, next_k_items=config['next_k_items'], plot=sp)
    print('LSTM Predictions Finished!')

    seqs_test2 = create_dataset_lstm_global(lstm_preds_tst,
                                            arima_preds_tst,
                                            data[config['features']].values[(trn_samples + config['next_k_items']):,
                                            config['features'].index('Close')])

    lstm_global_preds = lstm_global.predict(seqs_test2, plot=sp)

    plt.show()

    print('Finished')




main_other()
# x = np.sin(np.linspace(0, 3 * 2 * np.pi, 3 * 360)) + np.random.normal(0, 0.011, 3 * 360)
#
# plt.plot(x)
#
# arima = ARIMA(x, order=(50,1,0))
#
# arima_fit = arima.fit(disp=0)
#
# plt.plot(list(range(3 * 360 + 1, 3 * 360 + 200 + 1)), arima_fit.forecast(200)[0], 'r')
# plt.show()