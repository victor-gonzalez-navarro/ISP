import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.client import device_lib
from news.news_utils import read_news_all, prune_news, add_labels
from news.common import SP500_PATH, APPLE_PATH, PRODUCTS_PATH, NEWS_PATH, WINDOWS_SIZES, TRAIN
from src.news.trainers.train_model_LSTM_Regressor import rebalance

from news.stock_utils import load_all
from sklearn.linear_model import LinearRegression

from tqdm import tqdm
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

    # --- Train ---
    for i, window_size in enumerate(tqdm(WINDOWS_SIZES)):
        corpus = [' '.join(article[ATTRIBUTE]) for article in news]
        vectorizer = CountVectorizer(max_features=MAX_FEATURES)

        X = vectorizer.fit_transform(corpus)
        Y = np.array([article['windows'][window_size] for article in news])

        X_train, Y_train, X_test, Y_test = gen_training_tree(X, Y, test_size=(1.0 - TRAIN))

        clf = tree.DecisionTreeRegressor()
        clf = clf.fit(X_train, Y_train)
        predicted_y = clf.predict(X_test)

        # -- Subplot
        x = np.array(Y_test)
        y = np.array(predicted_y)
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        r_sq = model.score(x.reshape(-1, 1), y)
        y_line = model.predict(x.reshape(-1, 1))

        plt.subplot(2, 2, i + 1)
        plt.scatter(x, y, alpha=0.5)
        plt.plot(x, y_line, color='orange', linewidth=3, alpha=0.8)
        plt.xlabel('Real')
        plt.ylabel('Predicted')
        plt.gca().set_title('Pearson Corr Coef {:.3f}'.format(np.corrcoef(x, y)[0, 1]))
    plt.show()


if __name__ == '__main__':
    register_matplotlib_converters()
    print('Using ', device_lib.list_local_devices()[-1].name)
    main()
