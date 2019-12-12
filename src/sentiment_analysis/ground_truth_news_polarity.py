import json
import re
import csv
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import hsv_to_rgb
from pandas.plotting import register_matplotlib_converters

from utils import smooth

FILE_PATH = Path(__file__).resolve().parents[0]
IN_PATH = (FILE_PATH / '../../data/news_polarity.csv').resolve()
STOCK_PATH = (FILE_PATH / '../../data/sp500.csv').resolve()
SMOOTH_SIZE = 3
PLOT = False
PERIOD = 7  # Days before day d, where averages are taken
WINDOWS_SIZES = (2, 10, 20, 30)
WINDOW_PLOT_SIZE = WINDOWS_SIZES[1]


def read_news(path: Path):
    news = []
    with path.open('r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            item = {}
            item['date'] = datetime.strptime(row[0], '%m/%d/%y')
            item['polarity'] = float(row[1])
            news.append(item)
    return news


def read_stocks(path: Path, start=None, end=None, normalize=True):
    """ Read stocks values from csv pick those between start and end """
    data = pd.read_csv(path)
    data['Date'] = data['Date'].apply(lambda x: datetime.strptime(x, '%d-%m-%y'))
    x = data['Date'].tolist()
    y = data['Open'].tolist()
    if start is not None and end is not None:
        idx_start = next(i for i, v in enumerate(x) if v >= start)
        idx_end = next(i for i, v in enumerate(x) if v >= end)
        x = x[idx_start:idx_end]
        y = y[idx_start:idx_end]
    if normalize:
        y = (np.array(y) - np.min(y)) / (np.max(y) - np.min(y))
    return x, y


def findy(x, y, d):
    idx = 0
    while x[idx] <= d:
        idx += 1
    return y[idx]


def select_color(value):
    value = np.clip(value, -0.08, +0.08)
    h = ((value + 0.08) / 0.16) * 120.0 / 360.0
    s = 1.0
    v = 1.0
    return hsv_to_rgb((h, s, v))


def plot_ground_truth_per_article(x, y, news, window_size=WINDOW_PLOT_SIZE, multiplier=1.0):
    colors = [select_color(article['polarity']) for article in news]
    news_x = [article['date'] for article in news]
    news_y = [findy(x, y, article['date']) for article in news]

    plt.figure()
    plt.plot(x, y, alpha=0.8, label='%s Stock', zorder=0)
    plt.scatter(news_x, news_y, alpha=0.8, s=20, c=colors, zorder=1)
    plt.xticks(rotation=30)
    plt.legend()
    plt.show()


def prune_news(news, max_date):
    return [article for article in news if article['date'] < max_date]


def add_labels(x, y, news, prev_window=0, next_windows=WINDOWS_SIZES, multiplier=1, limits=None, rebalance=False, rebalance_limit=0.05):
    assert all(news[i]['date'] <= news[i + 1]['date'] for i in range(len(news) - 1))  # Assert sorted

    offset = 0
    for article in news:
        idx = next(i for i, v in enumerate(x[offset:]) if v >= article['date']) + offset
        start = max(0, idx - prev_window)

        article['windows'] = next_windows
        for next_window in next_windows:
            end = min(len(y) - 1, idx + next_window)
            article[next_window] = (y[end] / y[start]) * multiplier

            if limits and (article[next_window] - multiplier) > limits:
                article[next_window] = multiplier + limits
            if limits and (multiplier - article[next_window]) > limits:
                article[next_window] = multiplier - limits

        offset = idx

    return news


def add_labels_classification(x, y, news, prev_window=0, next_windows=WINDOWS_SIZES):
    assert all(news[i]['date'] <= news[i + 1]['date'] for i in range(len(news) - 1))  # Assert sorted

    offset = 0
    for article in news:
        idx = next(i for i, v in enumerate(x[offset:]) if v >= article['date']) + offset
        start = max(0, idx - prev_window)

        article['windows'] = next_windows
        for next_window in next_windows:
            end = min(len(y) - 1, idx + next_window)
            val = y[end] / y[start]
            label = 0

            # if val > 1.0:
            #     if val - 1.0 < 0.005: label = 4
            #     elif val - 1.0 < 0.02: label = 5
            #     else: label = 6
            # else:
            #     if 1.0 - val < 0.005: label = 3
            #     elif 1.0 - val < 0.02: label = 2
            #     else: label = 1

            if val >= 1.0:
                label = 1
            else:
                label = 0

            article[next_window] = label
        offset = idx

    return news


def get_news(news, begin: datetime, end: datetime):
    return [article for article in news if begin <= article['date'] <= end]


def generate_daily_values(news, begin: datetime, end: datetime):
    gt = []

    delta = timedelta(days=1)
    while begin <= end:
        selection = get_news(news, begin=begin - timedelta(PERIOD), end=begin)  # TODO optimize

        weighted_sum = 0
        mx = (PERIOD + 1) * (PERIOD + 2) / 2
        for i in range(PERIOD + 1):
            sub_selection = get_news(selection, begin=begin - timedelta(i), end=begin)
            weighted_sum += (i + 1) * sub_selection[0]['date'] / mx

        begin += delta


def main():
    news = read_news(IN_PATH)
    x, y = read_stocks(STOCK_PATH)
    smooth_y = smooth(y, half_window=SMOOTH_SIZE)
    news = prune_news(news, max_date=x[-1])
    plot_ground_truth_per_article(x, smooth_y, news)


if __name__ == '__main__':
    register_matplotlib_converters()
    main()
