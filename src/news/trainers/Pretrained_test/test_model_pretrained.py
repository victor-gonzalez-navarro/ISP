import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.client import device_lib
from news.news_utils import read_news_all, prune_news, add_labels
from news.common import SP500_PATH, APPLE_PATH, PRODUCTS_PATH, NEWS_PATH, WINDOWS_SIZES, TRAIN
from src.news.trainers.train_model_LSTM_Regressor import rebalance
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from news.stock_utils import load_all
from sklearn.linear_model import LinearRegression

from tqdm import tqdm

from pandas.plotting import register_matplotlib_converters

MAX_FEATURES = 2000
SMOOTH_SIZE = 10
ATTRIBUTE = 'word_vector'
DDBB = ['nyt']


def main():
    # --- Load ---
    sp500_x, apple_x, \
    sp500_y, apple_y, \
    percent_sp500_y, percent_apple_y, \
    correct_apple_y, diff, items = load_all(SMOOTH_SIZE, PRODUCTS_PATH, SP500_PATH, APPLE_PATH)
    news = read_news_all(NEWS_PATH, date_attribute='date', date_format='%Y-%m-%d %H:%M:%S', sort_date=True, databases=DDBB)

    apple_y = correct_apple_y
    # --- Prepare ---
    nltk.download('vader_lexicon')
    news = prune_news(news, max_date=apple_x[-1])
    news = add_labels(apple_x, apple_y, news)
    news = rebalance(news)

    # --- Train ---
    for k, window_size in enumerate(WINDOWS_SIZES):
        Y_train = []
        predicted_y = []
        for article in tqdm(news, desc='window size %d' % window_size):
            sid = SentimentIntensityAnalyzer()
            predicted_y.append(sid.polarity_scores(article['content'])['compound'])
            Y_train.append(article['windows'][window_size])

        diff = [(predicted_y[i] - Y_train[i]) for i in range(len(Y_train))]

        # -- Subplot
        x = np.array(predicted_y)
        y = np.array(Y_train)
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        r_sq = model.score(x.reshape(-1, 1), y)
        y_line = model.predict(x.reshape(-1, 1))

        plt.subplot(2, 2, k + 1)
        plt.scatter(predicted_y, Y_train, c=diff, alpha=0.8)
        plt.plot(x, y_line, color='orange', linewidth=3, alpha=0.8)
        plt.title('WS: ' + str(window_size))
        plt.xlabel('Score')
        plt.ylabel('Real')
        plt.gca().set_title('Pearson Corr Coef {:.3f}'.format(np.corrcoef(x, y)[0, 1]))
    plt.show()


if __name__ == '__main__':
    register_matplotlib_converters()
    print('Using ', device_lib.list_local_devices()[-1].name)
    main()
