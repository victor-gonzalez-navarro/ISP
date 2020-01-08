import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from tensorflow.python.client import device_lib
from src.news.news_utils import read_news_all, prune_news, add_labels
from src.news.common import SP500_PATH, APPLE_PATH, PRODUCTS_PATH, NEWS_PATH, WINDOWS_SIZES, TRAIN
from src.news.trainers.train_model_LSTM_Regressor import rebalance
from nltk.stem.porter import PorterStemmer

from src.news.stock_utils import load_all
from collections import defaultdict
from pandas.plotting import register_matplotlib_converters

MAX_FEATURES = 2000
SMOOTH_SIZE = 10
ATTRIBUTE = 'word_vector'
DDBB = ['nyt']
WINDOW_SIZE = WINDOWS_SIZES[2]  # 10
STEM = False

def find_words(news, window_size, condition):
    stemmer = PorterStemmer()
    global_dict = defaultdict(lambda: 0)
    for article in news:
        if condition(article['windows'][window_size]):
            article_dict = []
            for token in article['word_vector']:
                if STEM:
                    token = stemmer.stem(token)
                article_dict.append(token)

            for k in article_dict:
                global_dict[k] += 1

    average_dict = {}
    for k, v in global_dict.items():
        average_dict[k] = np.average(v)

    # Sorted dict
    sorted_dict = {k: v for k, v in reversed(sorted(average_dict.items(), key=lambda item: item[1]))}

    return sorted_dict


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

    # Count total positive and total negative words
    length_pos = 0
    length_neg = 0
    for article in news:
        if article['windows'][WINDOW_SIZE] > 1.0:
            length_pos += len(article['word_vector'])
        if article['windows'][WINDOW_SIZE] < 1.0:
            length_neg += len(article['word_vector'])


    # Find positive and negative word dictionaries
    sorted_dict_pos = find_words(news, WINDOW_SIZE, lambda k: k > 1.0)
    sorted_dict_neg = find_words(news, WINDOW_SIZE, lambda k: k < 1.0)

    # Sum both dictionaries
    total = defaultdict(lambda: 0)
    for k, v in sorted_dict_pos.items(): total[k] += v
    for k, v in sorted_dict_neg.items(): total[k] += v

    # Sort dictionary and word list in order of frequency
    sorted_total = {k: v for k, v in reversed(sorted(total.items(), key=lambda item: item[1]))}
    words = []
    for k, v in sorted_total.items():
        if v > 1.0:
            words.append(k)

    # Show words and their frequency
    step = 15
    i = step
    while i < len(words):
        for word in words[(i - 15):i]:
            print('{:15s}'.format(word[:20]), end=' ')
        print()
        for word in words[(i - 15):i]:
            pos = (sorted_dict_pos[word] if word in sorted_dict_pos else 0.0) * 100 / length_pos
            neg = (sorted_dict_neg[word] if word in sorted_dict_neg else 0.0) * 100 / length_neg
            print('{:.2f}|{:<.2f}      '.format(pos, neg), end=' ')
        print()
        print()
        i += step


if __name__ == '__main__':
    register_matplotlib_converters()
    print('Using ', device_lib.list_local_devices()[-1].name)
    main()
