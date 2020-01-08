import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.client import device_lib
from src.news.news_utils import read_news_all, prune_news, add_labels
from src.news.common import SP500_PATH, APPLE_PATH, PRODUCTS_PATH, NEWS_PATH, WINDOWS_SIZES, TRAIN
from src.news.trainers.train_model_LSTM_Regressor import rebalance
from src.news.trainers.train_model_LSTM_Classifier import labels_to_binary

from sklearn.metrics import confusion_matrix
from src.news.stock_utils import load_all

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import tree

from pandas.plotting import register_matplotlib_converters

MAX_FEATURES = 2000
SMOOTH_SIZE = 10
ATTRIBUTE = 'word_vector'
DDBB = ['nyt']


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


def main():
    # LSTM / Regressor
    # --- Load ---
    sp500_x, apple_x, \
    sp500_y, apple_y, \
    percent_sp500_y, percent_apple_y, \
    correct_apple_y, diff, items = load_all(SMOOTH_SIZE, PRODUCTS_PATH, SP500_PATH, APPLE_PATH)
    news = read_news_all(NEWS_PATH, date_attribute='date', date_format='%Y-%m-%d %H:%M:%S', sort_date=True, databases=DDBB)

    apple_y = correct_apple_y
    # --- Prepare ---
    news = prune_news(news, max_date=apple_x[-1])
    news = add_labels(apple_x, apple_y, news)
    news = rebalance(news)
    news = labels_to_binary(news)

    # --- Train ---
    for i, window_size in enumerate(WINDOWS_SIZES):
        corpus = [' '.join(article[ATTRIBUTE]) for article in news]
        vectorizer = CountVectorizer(max_features=MAX_FEATURES)

        X = vectorizer.fit_transform(corpus)
        Y = np.array([article['windows'][window_size] for article in news])

        X_train, Y_train, X_test, Y_test = gen_training_tree(X, Y, test_size=(1.0 - TRAIN))

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X_train, Y_train)
        predicted_y = clf.predict(X_test)

        y_true = []
        y_pred = []
        for i in range(len(predicted_y)):
            y_true.append('pos' if Y_test[i] == 1.0 else 'neg')
            y_pred.append('pos' if predicted_y[i] == 1.0 else 'neg')
        conf = confusion_matrix(y_true, y_pred, labels=['pos', 'neg'])
        print('Window size', window_size, ' BOW classifier, confusion matrix:\n', conf)
    plt.show()


if __name__ == '__main__':
    register_matplotlib_converters()
    print('Using ', device_lib.list_local_devices()[-1].name)
    main()
