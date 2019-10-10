import json
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

    m = Model('Model_{date:%Y-%m-%d_%H_%M_%S}.txt'.format(date=datetime.datetime.now()), config)
    m.buildLayers()
    m.fit(gen, len(seqs_train))

def main_best_model():
    with open('../config/params.json', 'r') as read_file:
        config = json.load(read_file)

    data = read_data('../data/sp500.csv')
    seqs_train, seqs_test = preprocess(data, config)

    gen = create_gen(seqs_train, config['batch_size'])

main()