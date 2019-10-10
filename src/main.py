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

    m = Model('Model_{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now()), config)
    m.buildLayers()
    m.fit(gen, len(seqs_train))

    m.predict(seqs_test, next_k_items=config['next_k_items'])

def main_best_model():
    with open('../config/params.json', 'r') as read_file:
        config = json.load(read_file)

    data = read_data('../data/sp500.csv')
    seqs_train, seqs_test = preprocess(data, config)

    fig = plt.figure()
    sp = fig.add_subplot(1, 1, 1)
    sp.plot(list(range(len(seqs_test))), [y for x, y in seqs_test], linewidth=0.5)

    m = Model(config, model_path='../models/Model_2019-10-10_14_40_56.txt')
    m.predict(seqs_test, next_k_items=config['next_k_items'], plot=sp)

    plt.show()

    print('Finished')


main_best_model()