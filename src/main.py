import json
import warnings
import pickle
import matplotlib.pyplot as plt

from lstm_global import LSTM_Global
from mlp import MLP_Network
from model2 import Model2
from utils import preprocess, preprocess_all, preprocess_all_extra, color_gen, predict_arima, \
    create_dataset_lstm_global, create_dataset_lstm_global2, predict_arima_all, create_dataset_reg, rmse, \
    create_dataset_mlp
from utils import read_data
from utils import create_gen
from model import Model
from sklearn.ensemble import RandomForestRegressor
import datetime
import numpy as np
import pandas as pd
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
    # lstm_preds_trn = m.predict(seqs_train, next_k_items=1)  # config['next_k_items'])
    lstm_preds_trn = m.predict_all(seqs_train, next_k_items=config['next_k_items'])

    print('Fitting ARIMA model...')
    # arima_preds_trn = predict_arima(seqs_train, config, next_k_items=1)  # config['next_k_items'])
    arima_preds_trn = predict_arima_all(seqs_train, config, next_k_items=config['next_k_items'])

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

    m = Model2(config, model_path='../models/Model2_2019-11-09_19_00_15')

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

    # trn_samples = round(config['train_test_split'] * data.shape[0])
    seqs_train2, y_train = create_dataset_reg(lstm_preds_trn[:-config['next_k_items']],
                                              arima_preds_trn[:-config['next_k_items']],
                                              [seq[1][config['features'].index('Close')] for seq in seqs_train])
    # y_train = [seq[1][config['features'].index('Close')] for seq in seqs_train]

    model_rf = RandomForestRegressor(n_estimators=config['rf_estimators'], random_state=100)
    model_rf.fit(seqs_train2, y_train)

    seqs_train2 = create_dataset_mlp(lstm_preds_trn[:-config['next_k_items']],
                                     arima_preds_trn[:-config['next_k_items']],
                                     [seq[1][config['features'].index('Close')] for seq in seqs_train])

    trn_predictions = []
    for seq in seqs_train2:
        trn_reg_preds = model_rf.predict(seq[0])
        trn_predictions.append(trn_reg_preds)

    y_mlp = np.array([seq[1] for seq in seqs_train2])

    trn_dates = data.Date.iloc[config['next_k_items'] - 1:round(data.shape[0] * config['train_test_split']) - 31].values
    tst_dates_ixs = np.arange(round(data.shape[0] * config['train_test_split']) + config['next_k_items'] - 1, data.shape[0], 30)
    tst_dates = data.Date.iloc[tst_dates_ixs].values

    with open('../data/news_predicted.json') as f:
        news_pred_json = json.load(f)
    news_pred = [[row['date']] + row['predicted_labels'] for row in news_pred_json]
    news_pred = pd.DataFrame(news_pred, columns=['Date', 'T_2', 'T_10', 'T_20', 'T_30'])
    news_pred.Date = pd.to_datetime(news_pred.Date)
    news_pred.Date = news_pred.Date.apply(lambda s: date_string(s))

    valid_trn_dates = np.isin(trn_dates, news_pred.Date.values)
    valid_tst_dates = np.isin(tst_dates, news_pred.Date.values)

    valid_y_mlp = y_mlp[valid_trn_dates, :]
    valid_trn_preds = []
    for i in range(len(trn_predictions)):
        if valid_trn_dates[i]:
            valid_trn_preds.append(np.concatenate((trn_predictions[i], news_pred[news_pred.Date == trn_dates[i]].values[0, 1:])))

    # mlp_gen = create_gen(list(zip(trn_predictions, y_mlp)), config['mlp']['batch_size'])

    mlp = MLP_Network(config, name='MLP_{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now()))
    mlp.buildLayers(len(valid_trn_preds[0]))
    mlp.fit(valid_trn_preds, valid_y_mlp)
    # mlp.fit2(mlp_gen, len(trn_predictions))

    #gen2 = create_gen(seqs_train2, config['batch_size'])

    #lstm_global = LSTM_Global(config, name='LSTM_Global_{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now()))
    #lstm_global.buildLayers()
    #print('Fitting LSTM Global model...')
    #lstm_global.fit(gen2, len(seqs_train2))

    print('Starting ARIMA Predictions...')
    arima_preds_tst = predict_arima(seqs_test, config, next_k_items=config['next_k_items'])  #, plot=sp)
    print('ARIMA Predictions Finished!')

    print('Starting LSTM Predictions...')
    lstm_preds_tst = m.predict(seqs_test, next_k_items=config['next_k_items'])  #, plot=sp)
    print('LSTM Predictions Finished!')

    seqs_test2 = create_dataset_lstm_global(lstm_preds_tst,
                                            arima_preds_tst,
                                            [seq[1][config['features'].index('Close')] for seq in seqs_test])

    #lstm_global_preds = lstm_global.predict(seqs_test2, plot=sp)
    plot = True
    end = 0
    color = color_gen()

    predictions = []
    for seq_ix in range(len(seqs_test2)):
        reg_preds = model_rf.predict(seqs_test2[seq_ix][0])

        if valid_tst_dates[seq_ix]:
            mlp_preds = mlp.predict(np.concatenate((reg_preds,news_pred[news_pred.Date == tst_dates[seq_ix]].
                                                    values[0, 1:])).reshape(1, -1))[0, :]
            predictions.append(mlp_preds)
        else:
            predictions.append(reg_preds)

        fft = np.fft.fft(predictions[-1])
        aux_fft = np.copy(fft)
        aux_fft[config['fourier_coef_2']:-config['fourier_coef_2']] = 0
        ifft = np.fft.ifft(aux_fft).real.reshape((len(aux_fft), 1))
        predictions[-1] = ifft

        if plot:
            start = end
            end = start + len(predictions[-1])
            sp.plot(list(range(start, end)), predictions[-1], next(color) + '*-', markersize=2, linewidth=0.5)

    plt.show()

    print(rmse(np.array([val for preds in predictions for val in preds]), np.array([val for seq in seqs_test2 for val in seq[1]])))

    print('Finished')


def date_string(s):
    t = datetime.datetime.strftime(s, '%d-%m-%Y')
    return t[:-4] + t[-2:]


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