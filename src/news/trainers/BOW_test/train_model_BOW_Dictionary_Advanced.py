import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.python.client import device_lib
from src.news.news_utils import read_news_all, prune_news, add_labels
from src.news.common import SP500_PATH, APPLE_PATH, PRODUCTS_PATH, NEWS_PATH, WINDOWS_SIZES, NEGATIVE_PATH, POSITIVE_PATH, TRAIN
from src.news.trainers.train_model_LSTM_Regressor import rebalance
from sklearn.linear_model import LinearRegression

from src.news.stock_utils import load_all
from pandas.plotting import register_matplotlib_converters
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

MAX_FEATURES = 2000
SMOOTH_SIZE = 10
ATTRIBUTE = 'word_vector'
DDBB = ['nyt']
WINDOW_SIZE = WINDOWS_SIZES[2]  # 10


def load_words(path: Path):
    words = []
    with path.open('r') as f:
        for line in f:
            word = line.strip().lower()
            if len(word) > 0:
                words.append(word)

    return words


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
    negative = set(load_words(NEGATIVE_PATH))
    positive = set(load_words(POSITIVE_PATH))

    # --- Train ---
    for k, window_size in enumerate(tqdm(WINDOWS_SIZES)):
        xx = []
        yy = []
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
            yy.append(article['windows'][window_size])

        # -- Subplot
        x = np.array(xx)
        y = np.array(yy)
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        r_sq = model.score(x.reshape(-1, 1), y)
        y_line = model.predict(x.reshape(-1, 1))

        plt.subplot(2, 2, k + 1)
        plt.scatter(xx, yy, alpha=0.5)
        plt.plot(x, y_line, color='orange', linewidth=3, alpha=0.8)
        plt.xlabel('Score')
        plt.ylabel('Label')
        plt.gca().set_title('Pearson Corr Coef {:.3f}'.format(np.corrcoef(x, y)[0, 1]))
    plt.show()


if __name__ == '__main__':
    register_matplotlib_converters()
    print('Using ', device_lib.list_local_devices()[-1].name)
    main()
