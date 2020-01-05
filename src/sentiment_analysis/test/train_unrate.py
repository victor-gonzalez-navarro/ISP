from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

from src.sentiment_analysis.ground_truth import read_news, read_stocks
from src.sentiment_analysis.test.common import *

FILE_PATH = Path(__file__).resolve().parents[0]
IN_PATH = (FILE_PATH / '../../../data/preprocessed.json').resolve()
STOCK_PATH = (FILE_PATH / '../../../data/sp500.csv').resolve()
PLOT = False


def derivative(x, y):
    assert len(x) == len(y)
    return x[:-1], [y[i + 1] - y[i] for i in range(len(y) - 1)]


def smooth(y, half_window=10):
    sy = [0] * len(y)
    for i in range(len(y)):
        begin = max(0, i - half_window)
        end = min(len(y) - 1, i + half_window)
        sy[i] = np.average(y[begin:end])
    return sy


def normalize(y):
    return (np.array(y) - np.min(y)) / (np.max(y) - np.min(y))


def to_percent(y):
    for i in range(1, len(y)):
        print(y[i],  y[i - 1], y[i] / y[i - 1])
    return [(y[i] + 0.0001) / (y[i - 1] + 0.0001) for i in range(1, len(y))] + [1.0]


def read_unrate(path: Path, start=None, end=None, normalize=True):
    """ Read stocks values from csv pick those between start and end """
    data = pd.read_csv(path)
    data['Date'] = data['Date'].apply(lambda x: datetime.strptime(x, '%d-%m-%y'))
    x = data['Date'].tolist()
    y = data['Unrate'].tolist()
    if start is not None and end is not None:
        idx_start = next(i for i, v in enumerate(x) if v >= start)
        idx_end = next(i for i, v in enumerate(x) if v >= end)
        x = x[idx_start:idx_end]
        y = y[idx_start:idx_end]
    if normalize:
        y = (np.array(y) - np.min(y)) / (np.max(y) - np.min(y))
    return x, y


def cut(x, y, begin, end):
    y = [j for i, j in enumerate(y) if begin < x[i] < end]
    x = [i for i in x if begin < i < end]
    return x, y


def main():
    # news = read_news(IN_PATH)

    sp_x, sp_y = read_stocks(STOCK_PATH, normalize=False)
    aapl_x, aapl_y = read_stocks((FILE_PATH / '../../../data/AAPL.csv').resolve())
    x, y = read_unrate((FILE_PATH / '../../../data/unrate.csv').resolve())
    sp_y = smooth(sp_y, half_window=10)
    aapl_y = smooth(aapl_y, half_window=10)
    y = smooth(y, half_window=5)

    dx, dy = derivative(x, y)
    dy = smooth(dy, half_window=5)
    dy = np.array(dy) * 10

    plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(sp_x, normalize(sp_y), color='red')
    plt.plot(aapl_x, normalize(aapl_y), color='blue')

    ax2 = plt.subplot(212, sharex=ax1)
    plt.plot(x, y, color='orange')
    plt.plot(dx, dy, color='brown')
    plt.axhline(0)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    register_matplotlib_converters()
    main()
