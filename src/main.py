import json
import matplotlib.pyplot as plt
from utils import preprocess
from utils import read_data
from utils import create_gen
from model import Model
import datetime


def main_new_model():
    with open('../config/params.json', 'r') as read_file:
        config = json.load(read_file)

    data = read_data('../data/sp500.csv')
    seqs_train, seqs_test = preprocess(data, config)

    gen = create_gen(seqs_train, config['batch_size'])

    fig = plt.figure()
    sp = fig.add_subplot(1, 1, 1)
    sp.plot(list(range(len(seqs_test))), [y for x, y in seqs_test], linewidth=0.5)

    m = Model(config, name='Model_{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now()))
    m.buildLayers()
    m.fit(gen, len(seqs_train))

    m.predict(seqs_test, next_k_items=config['next_k_items'], plot=sp)

    plt.show()

def main_best_model():
    with open('../config/params.json', 'r') as read_file:
        config = json.load(read_file)

    data = read_data('../data/sp500.csv')
    seqs_train, seqs_test = preprocess(data, config)

    gen = create_gen(seqs_train, config['batch_size'])

    yii = next(gen)
    a = yii[0][1, :, :]
    a2 = yii[0][6, :, :]
    yii2 = next(gen)
    b = yii2[0][1, :, :]
    b2 = yii2[0][6, :, :]
    yii3 = next(gen)
    c = yii3[0][1, :, :]
    c2 = yii3[0][6, :, :]

    fig = plt.figure()
    sp = fig.add_subplot(1, 1, 1)
    sp.plot(list(range(len(seqs_test))), [y for x, y in seqs_test], linewidth=0.5)

    m = Model(config, model_path='../models/11102019-120649-e2.h5')
    m.predict(seqs_test, next_k_items=config['next_k_items'], plot=sp)

    plt.show()

    print('Finished')


main_new_model()
