import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datetime import datetime
from pathlib import Path
import random
import json

from sentiment_analysis.w2v_models import GensimModel
import matplotlib.pyplot as plt
import numpy as np
from utils import smooth
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from matplotlib.colors import hsv_to_rgb
from tqdm import tqdm
from src.sentiment_analysis.ground_truth import read_news, read_stocks, prune_news, add_labels

from sentiment_analysis.NNModel import NNModel

from pandas.plotting import register_matplotlib_converters


FILE_PATH = Path(__file__).resolve().parents[0]
IN_PATH = (FILE_PATH / '../../../data/preprocessed_content.json').resolve()
STOCK_PATH = (FILE_PATH / '../../../data/AAPL.csv').resolve()
MODEL_PATH = (FILE_PATH / '../../../models/news_model.h5').resolve()

BEGIN_TIME = datetime(year=2000, month=1, day=1)
END_TIME = datetime(year=2018, month=1, day=4)

MAX_INPUT_LEN = 80
W2V_MODEL_NAME = 'glove-wiki-gigaword-50'
SMOOTH_SIZE = 3
PLOT = False
PERIOD = 7  # Days before day d, where averages are taken
WINDOWS_SIZES = (2, 10, 20, 30)
WINDOW_PLOT_SIZE = WINDOWS_SIZES[1]


def init():
    register_matplotlib_converters()
    print('Using ', device_lib.list_local_devices()[-1].name)


def rebalance(news, rebalance_limit=0.05):
    pos_samples = []
    neg_samples = []

    for article in news:
        avg_val = np.average([article[next_window] for next_window in article['windows']])

        if avg_val >= 1.0:
            pos_samples.append(article)
        else:
            neg_samples.append(article)

    pos_percentage = len(pos_samples) / len(news)
    neg_percentage = len(neg_samples) / len(news)
    print('Positive samples: {:.1f}% ({:d} samples)'.format(pos_percentage * 100.0, len(pos_samples)))
    print('Negative samples: {:.1f}% ({:d} samples)'.format(neg_percentage * 100.0, len(neg_samples)))
    if rebalance and abs(pos_percentage - neg_percentage) >= rebalance_limit:
        pop_size = min(len(pos_samples), len(neg_samples))
        rebalanced_news = random.sample(pos_samples, k=pop_size) + random.sample(neg_samples, k=pop_size)
        print('Rebalanced to 50/50 ({:d} samples each)'.format(pop_size))
        return sorted(rebalanced_news, key=lambda x: x['date'])

    return news


def gen_training(news, test_size, max_input_len, window_size, has_embeddings=True):
    # data = [(article['word_indexes'], article[window_size] for article in news]
    # data = [(article['word_indexes'], 0 if article[WINDOWS_SIZES[0]] < 1.0 else 1.0) for article in news]
    # train, test = train_test_split(data, test_size=test_size)
    train = [(article['word_indexes'], article[window_size]) for article in news[:test_size]]
    test = [(article['word_indexes'], article[window_size]) for article in news[test_size:]]
    print('Last of train is ', news[:test_size][-1]['date'])
    print('First of test is ', news[test_size:][-1]['date'])
    x_train, y_train = list(zip(*train))
    x_test, y_test = list(zip(*test))
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    if has_embeddings:
        x_train = sequence.pad_sequences(x_train, dtype=object, maxlen=max_input_len)
        x_test = sequence.pad_sequences(x_test, dtype=object, maxlen=max_input_len)
    else:
        x_train = sequence.pad_sequences(x_train, maxlen=max_input_len)
        x_test = sequence.pad_sequences(x_test, maxlen=max_input_len)
    return x_train, y_train, x_test, y_test


def to_percent(y):
    return [(y[i] + 0.0001) / (y[i - 1] + 0.0001) for i in range(1, len(y))] + [1.0]


def normalize(y):
    return (np.array(y) - np.min(y)) / (np.max(y) - np.min(y))


def main():
    init()

    # sp_x, sp_y = read_stocks((FILE_PATH / '../../../data/sp500.csv').resolve(), normalize=False)
    aapl_x, aapl_y = read_stocks((FILE_PATH / '../../../data/AAPL.csv').resolve(), normalize=False)
    news = read_news(IN_PATH)
    y = smooth(aapl_y, half_window=10)
    x = aapl_x

    # sp_y = smooth(sp_y, half_window=10)
    # aapl_y = smooth(aapl_y, half_window=10)
    # start_aapl_y = normalize(aapl_y)[0]
    #
    # # smooth_y = smooth(y, half_window=SMOOTH_SIZE)
    # #  plot_ground_truth_per_article(x, smooth_y, news)
    #
    # sp_y = to_percent(sp_y)
    # aapl_y = to_percent(aapl_y)
    # sp_y = smooth(sp_y, half_window=10)
    # aapl_y = smooth(aapl_y, half_window=10)
    #
    # y = [start_aapl_y]
    # for i in range(len(aapl_y)):
    #     d = abs(sp_y[i] - 1) * 2
    #     b = y[-1] * aapl_y[i]
    #     if sp_y[i] > 1.0:
    #         y.append(b * (1 - d))
    #     else:
    #         y.append(b * (1 + d))
    # x = aapl_x
    # y = y[:-1]

    news = prune_news(news, max_date=x[-1])
    news = add_labels(x, y, news)
    news = rebalance(news)

    # --- Train ---
    train_date = x[int(len(x) * 0.85)]
    test_size = next(i for i, article in enumerate(news) if article['date'] >= train_date)  # news[:test_size] is training
    word2vec_model = GensimModel(W2V_MODEL_NAME)  # BOWModel()

    models = [None] * len(WINDOWS_SIZES)
    for i, window_size in enumerate(WINDOWS_SIZES):
        x_train, y_train, x_test, y_test = gen_training(news, test_size=test_size, max_input_len=MAX_INPUT_LEN, window_size=window_size)

        models[i] = NNModel(word2vec_model.generate_embedding_layer(MAX_INPUT_LEN, trainable=False))
        models[i].compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
        #  models[i].load_weights(MODEL_PATH)
        history = models[i].fit(x_train, y_train, validation_data=(x_test, y_test), epochs=8, batch_size=64)
        models[i].save_weights('../../../models/news_model_' + str(window_size) + '_.h5')

    predictions = []
    X = [article['word_indexes'] for article in news]
    for model in tqdm(models):
        predictions.append(model.predict(sequence.pad_sequences(X, dtype=object, maxlen=MAX_INPUT_LEN)))

    modified_news = []
    for k, article in enumerate(tqdm(news)):
        article['real_labels'] = [article[window_size] for window_size in article['windows']]
        article['predicted_labels'] = [float(predictions[i][k][0]) for i, _ in enumerate(article['windows'])]
        modified_news.append(article)

    for i, _ in enumerate(WINDOWS_SIZES):
        for k, article in enumerate(tqdm(news)):
            plt.scatter(article['real_labels'][i], article['predicted_labels'][i])
        plt.show()

    with open('../../../data/news_predicted.json', 'w') as f:
        json.dump(modified_news, f, indent=4, default=str)

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss (mse)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # predicted_y = model.predict(x_train)
    # diff = [(predicted_y[i] - y_train[i]) for i in range(len(y_train))]
    # plt.figure()
    # plt.scatter(predicted_y, y_train, c=diff, alpha=0.8)
    # axes = plt.gca()
    # axes.set_xlim([-0.2, 0.2])
    # axes.set_ylim([-0.2, 0.2])
    # plt.show()


if __name__ == '__main__':
    main()
