import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datetime import datetime
from pathlib import Path
import random

from sentiment_analysis.w2v_models import GensimModel
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from matplotlib.colors import hsv_to_rgb
from utils import smooth
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
from src.sentiment_analysis.ground_truth import read_news, read_stocks, prune_news, add_labels_classification, add_labels, plot_ground_truth_per_article

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import _tree
from sklearn import tree
from sentiment_analysis.NNModel import NNModel

from pandas.plotting import register_matplotlib_converters


FILE_PATH = Path(__file__).resolve().parents[0]
IN_PATH = (FILE_PATH / '../../data/preprocessed_wsj6_.json').resolve()
NEGATIVE_PATH = (FILE_PATH / '../../data/negative_words.txt').resolve()
POSITIVE_PATH = (FILE_PATH / '../../data/positive_words.txt').resolve()
STOCK_PATH = (FILE_PATH / '../../data/AAPL.csv').resolve()
MODEL_PATH = (FILE_PATH / '../../models/news_model1.h5').resolve()

BEGIN_TIME = datetime(year=2007, month=1, day=1)
END_TIME = datetime(year=2008, month=1, day=4)

MAX_INPUT_LEN = 80
W2V_MODEL_NAME = 'glove-wiki-gigaword-100'
SMOOTH_SIZE = 3
PLOT = False
PERIOD = 7  # Days before day d, where averages are taken
WINDOWS_SIZES = (2, 10, 20, 30)
WINDOW_PLOT_SIZE = WINDOWS_SIZES[1]


def init():
    register_matplotlib_converters()
    print('Using ', device_lib.list_local_devices()[-1].name)


def rebalance(news, rebalance_limit=0.05, multiplier=1):
    pos_samples = []
    neg_samples = []

    for article in news:
        avg_val = np.average([article[next_window] for next_window in article['windows']])

        if avg_val >= 1.0 * multiplier:
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


def unison_shuffled_copies(a, b):
    assert a.shape[0] == b.shape[0]
    p = np.random.permutation(a.shape[0])
    return a[p], b[p]


def gen_training_tree(X, Y, test_size):
    assert X.shape[0] == Y.shape[0]

    X, Y = unison_shuffled_copies(X, Y)
    length = X.shape[0]
    train_size = length - int(test_size * length)

    X_train = X[:train_size]
    X_test = X[train_size:]
    Y_train = Y[:train_size]
    Y_test = Y[train_size:]
    return X_train, Y_train, X_test, Y_test


def gen_training(news, test_size, max_input_len, has_embeddings=True):
    data = [(article['word_indexes_big'], article[WINDOWS_SIZES[0]]) for article in news]
    train, test = train_test_split(data, test_size=test_size)
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


def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)


def load_words(path: Path):
    words = []
    with path.open('r') as f:
        for line in f:
            word = line.strip().lower()
            if len(word) > 0:
                words.append(word)

    return words


def print_lines(txt, line_length=90):
    while len(txt) > line_length:
        if txt[0] == ' ':
            txt = txt.strip()
        else:
            length = line_length + txt[line_length:].find(' ')
            print(txt[:length])
            txt = txt[length:]
    print(txt.strip())


def findy(x, y, d):
    idx = 0
    while x[idx] <= d:
        idx += 1
    return y[idx]


def select_color(score):
    value = np.clip(score, -20, 20)
    h = ((value + 20) / 40) * 120.0 / 360.0
    s = 1.0
    v = 1.0
    return hsv_to_rgb((h, s, v))


def plot_ground_truth_per_article2(x, y, news):
    colors = [select_color(article['score']) for article in news]
    news_x = [article['date'] for article in news]
    news_y = [findy(x, y, article['date']) for article in news]

    plt.figure()
    plt.plot(x, y, alpha=0.6, label='%s Stock', zorder=0)
    plt.scatter(news_x, news_y, alpha=0.8, s=8, c=colors, zorder=1)
    plt.xticks(rotation=30)
    plt.legend()
    plt.show()


def main():
    init()

    x, y = read_stocks(STOCK_PATH)
    news = read_news(IN_PATH)
    smooth_y = smooth(y, half_window=SMOOTH_SIZE)
    news = sorted(news, key=lambda x: x['date'])
    news = prune_news(news, max_date=x[-1])
    # news = add_labels_classification(x, smooth_y, news, next_windows=[4])  #, multiplier=100, limits=10)
    news = add_labels(x, smooth_y, news, multiplier=100, limits=10)
    news = rebalance(news, multiplier=100)
    # plot_ground_truth_per_article(x, smooth_y, news, multiplier=100)
    windows = news[0]['windows']

    # for article in news:
    #     if 'imagerendition' in article['word_vector']:
    #         print_lines(article['content'])
    #         exit()

    length_pos = 0
    for article in news:
        if article[windows[0]] > 100:
            length_pos += len(article['word_vector'])
    print(length_pos)
    length_neg = 0
    for article in news:
        if article[windows[0]] < 100:
            length_neg += len(article['word_vector'])
    print(length_neg)


    stemmer = PorterStemmer()
    # ----------------------------------------------------------------- POSITIVE
    global_dict = defaultdict(lambda: 0)
    for article in news:
        if article[windows[0]] > 100:
            article_dict = []
            for token in article['word_vector']:
                # token = stemmer.stem(token)
                article_dict.append(token)

            for k in article_dict:
                global_dict[k] += 1

    average_dict = {}
    for k, v in global_dict.items():
        average_dict[k] = np.average(v)

    sorted_dict_pos = {k: v for k, v in reversed(sorted(average_dict.items(), key=lambda item: item[1]))}
    ks = []
    vs = []
    for k, v in sorted_dict_pos.items():
        if v > 1.0:
            ks.append(k)
            vs.append(v)

    # for k in ks:
    #     print('{:9s}'.format(k), end=' ')
    # print()
    # for v in vs:
    #     print('{:<9d}'.format(int(v)), end=' ')
    # print()

    # ----------------------------------------------------------------- NEGATIVE
    global_dict = defaultdict(lambda: 0)
    for article in news:
        if article[windows[0]] < 100:
            article_dict = []
            for token in article['word_vector']:
                # token = stemmer.stem(token)
                article_dict.append(token)

            for k in article_dict:
                global_dict[k] += 1

    average_dict = {}
    for k, v in global_dict.items():
        average_dict[k] = np.average(v)

    sorted_dict_neg = {k: v for k, v in reversed(sorted(average_dict.items(), key=lambda item: item[1]))}
    ks = []
    vs = []
    for k, v in sorted_dict_neg.items():
        if v > 1.0:
            ks.append(k)
            vs.append(v)

    # for k in ks:
    #     print('{:9s}'.format(k), end=' ')
    # print()
    # for v in vs:
    #     print('{:<9d}'.format(int(v)), end=' ')
    # print()

    # -----------------------------------------------------------------
    total = defaultdict(lambda: 0)
    for k, v in sorted_dict_pos.items():
        total[k] += v
    for k, v in sorted_dict_neg.items():
        total[k] += v

    sorted_total = {k: v for k, v in reversed(sorted(total.items(), key=lambda item: item[1]))}
    words = []
    for k, v in sorted_total.items():
        if v > 1.0:
            words.append(k)

    i = 15
    while i < len(words):
        for word in words[(i-15):i]:
            print('{:15s}'.format(word[:20]), end=' ')
        print()
        for word in words[(i-15):i]:
            pos = (sorted_dict_pos[word] if word in sorted_dict_pos else 0.0) * 100 / length_pos
            neg = (sorted_dict_neg[word] if word in sorted_dict_neg else 0.0) * 100/ length_neg
            print('{:.2f}|{:<.2f}      '.format(pos, neg), end=' ')
        print()
        print()
        i += 15

    #exit()
    # -----------------------------------------------------------------
    negative = set(load_words(NEGATIVE_PATH))
    positive = set(load_words(POSITIVE_PATH))

    xx = []
    yy = []
    cc = []
    for i, article in enumerate(tqdm(news)):
        count_negative = 0
        count_positive = 0
        for token in article['word_vector']:
            if token in negative:
                count_negative += 1
            if token in positive:
                count_positive += 1

        news[i]['score'] = count_positive - count_negative
        xx.append(count_positive - count_negative)
        yy.append(article[windows[0]])
        cc.append(0)

    plt.figure()
    plt.scatter(xx, yy, c=cc, alpha=0.2)
    plt.xlabel('score')
    plt.ylabel('label')
    plt.show()

    plot_ground_truth_per_article2(x, y, news)


if __name__ == '__main__':
    main()
