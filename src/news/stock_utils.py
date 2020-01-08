from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from src.news.common import SP500_PATH, APPLE_PATH, PRODUCTS_PATH

PRODUCTS_TO_SHOW = ['iPhone']
SMOOTH_SIZE = 10


class Item:
    def __init__(self, date: datetime, products):
        self.date = date
        self.products = products


def normalize(y):
    return (np.array(y) - np.min(y)) / (np.max(y) - np.min(y))


def to_percent(y, epsilon=0.0001):
    """ Return the variation between consecutive values in percentage. Epsilon prevents inf math. Last value is fixed
     to 1.0 """
    return [(y[i] + epsilon) / (y[i - 1] + epsilon) for i in range(1, len(y))] + [1.0]


def derivative(x, y):
    """ Calculate the derivative. Note that both returned arrays have length - 1 """
    assert len(x) == len(y)
    return x[:-1], [y[i + 1] - y[i] for i in range(len(y) - 1)]


def smooth(y, half_window=10):
    """ Smooth an 1D array according to the window size using the average """
    sy = [0] * len(y)
    for i in range(len(y)):
        begin = max(0, i - half_window)
        end = min(len(y) - 1, i + half_window)
        sy[i] = np.average(y[begin:end])
    return sy


def read_csv(path: Path, date_attribute='Date', date_format='%d-%m-%y', start_date=None, end_date=None, sort_date=True):
    """ Read csv file and make some very basic processing on dates """
    data = pd.read_csv(path)
    cols = list(data.columns.values)

    # Transform date values from str
    if date_attribute in cols:
        data[date_attribute] = data[date_attribute].apply(lambda k: datetime.strptime(k, date_format))

    # Columns to lists
    outputs = [data[col].tolist() for col in cols]
    date_index = cols.index(date_attribute)

    # Trim according to start and end if date was found
    if start_date is not None and end_date is not None and date_attribute in cols:
        x = outputs[date_index]
        idx_start = next(i for i, v in enumerate(x) if v >= start_date)
        idx_end = next(i for i, v in enumerate(x) if v >= end_date)
        for i, output in enumerate(outputs):
            outputs[i] = output[idx_start:idx_end]

    # Sort according to date
    if date_attribute in cols and sort_date:
        join = zip(*outputs)
        join = sorted(join, key=lambda k: k[date_index])
        outputs = zip(*join)  # Unzip
        outputs = [list(row) for row in outputs]  # Convert back from tuple

    return outputs


def read_stocks(path: Path, start_date=None, end_date=None, normalize_data=True, sort_date=True):
    """ Read stocks values from csv. Pick those between start and end """
    date, open, high, low, close, volume = read_csv(path=path, start_date=start_date, end_date=end_date, sort_date=sort_date)
    x = date
    y = open
    return x, normalize(y) if normalize_data else y, volume


def read_products_launch(path: Path, start_date=None, end_date=None, delimiter=',', sort_date=True):
    """ Read launch dates for different products from csv. Pick those between start and end """
    dates, products = read_csv(path=path, start_date=start_date, end_date=end_date, sort_date=sort_date)
    return [Item(date, products=products[i].split(delimiter)) for i, date in enumerate(dates)]


def correct_aapl(apple_y, percent_apple_y, percent_sp500_y):
    """ Apply corrections to the AAPL stock value subtracting the influence by the SP500 """
    y = [apple_y[0]]
    for i in range(len(percent_apple_y) - 1):
        d = abs(percent_sp500_y[i] - 1) * 2
        b = y[-1] * percent_apple_y[i]
        if percent_sp500_y[i] > 1.0:
            y.append(b * (1 - d))
        else:
            y.append(b * (1 + d))

    y = [apple_y[0]]
    for i in range(len(percent_apple_y) - 1):
        if percent_sp500_y[i] > 1.0:
            d = (percent_sp500_y[i] - 1.0) * 2
            b = y[-1] * percent_apple_y[i]
            y.append(b * (1 - d))
        else:
            d = abs(percent_sp500_y[i] - 1) * 2
            b = y[-1] * percent_apple_y[i]
            y.append(b * (1 + d))

    return y


def __show_items(items, products_to_show, ax):
    """ Show in vertical lines the names of the items which contains any of the products in products to show """
    _, _, bottom, top = ax.axis()
    middle = (top + bottom) / 2

    for item in items:
        for product in products_to_show:
            if product in item.products:
                plt.axvline(x=item.date, zorder=-1, alpha=0.2, color='orange')
                plt.text(x=item.date, y=middle, s=','.join(item.products), fontsize=8, rotation=90, clip_on=True)
                break

    def on_ylims_change(_):
        _, _, _bottom, _top = ax.axis()
        _middle = (_top + _bottom) / 2
        for t in ax.texts:
            t.set_y(middle)

    ax.callbacks.connect('ylim_changed', on_ylims_change)


def load_all(smooth_size, products_path, sp500_path, apple_path):
    items = read_products_launch(products_path, sort_date=True)
    sp500_x, sp500_y, _ = read_stocks(sp500_path, normalize_data=False, sort_date=True)
    apple_x, apple_y, _ = read_stocks(apple_path, normalize_data=False, sort_date=True)

    # Smooth to soften changes
    sp500_y = smooth(sp500_y, half_window=smooth_size)
    apple_y = smooth(apple_y, half_window=smooth_size)

    # Percentage changes
    percent_sp500_y = to_percent(sp500_y)
    percent_apple_y = to_percent(apple_y)
    percent_sp500_y = smooth(percent_sp500_y, half_window=smooth_size)
    percent_apple_y = smooth(percent_apple_y, half_window=smooth_size)

    # Difference
    diff = (1.0 - np.array(percent_sp500_y)) * (np.array(percent_apple_y) - 1.0)
    diff = np.clip(diff, 0, 9999999)  # Positive values indicate a discordance between SP500 and AAPL

    # Correction
    correct_apple_x = apple_x
    correct_apple_y = correct_aapl(apple_y, percent_apple_y, percent_sp500_y)

    return sp500_x, apple_x, sp500_y, apple_y, percent_sp500_y, percent_apple_y, correct_apple_y, diff, items


def main():
    sp500_x, apple_x, \
    sp500_y, apple_y, \
    percent_sp500_y, percent_apple_y, \
    correct_apple_y, diff, items = load_all(SMOOTH_SIZE, PRODUCTS_PATH, SP500_PATH, APPLE_PATH)

    plt.figure()
    plt.suptitle('SP500 vs AAPL')

    ax1 = plt.subplot(221)
    plt.plot(sp500_x, normalize(sp500_y), color='red', label='S&P500')
    plt.plot(apple_x, normalize(apple_y), color='royalblue', label='APPL')
    plt.axhline(0.0, color='black', alpha=0.3, zorder=-1)
    plt.gca().set_title('Normalized open value')
    plt.legend()

    __show_items(items, PRODUCTS_TO_SHOW, ax1)

    plt.subplot(222, sharex=ax1)
    plt.plot(sp500_x, percent_sp500_y, color='red', alpha=0.8)
    plt.plot(apple_x, percent_apple_y, color='royalblue', alpha=0.8)
    plt.axhline(1.0, color='black', alpha=0.3, zorder=-1)
    plt.gca().set_title('Percentage change')

    plt.subplot(223, sharex=ax1)
    plt.plot(apple_x, normalize(diff), color='green')
    plt.axhline(0.0, color='black', alpha=0.3, zorder=-1)
    plt.gca().set_title('Percentage multiplication normalized')

    plt.subplot(224, sharex=ax1)
    plt.plot(apple_x, normalize(correct_apple_y), color='royalblue')
    plt.axhline(0.0, color='black', alpha=0.3, zorder=-1)
    plt.gca().set_title('Corrected AAPL')

    __show_items(items, PRODUCTS_TO_SHOW, ax1)

    plt.show()


if __name__ == '__main__':
    register_matplotlib_converters()
    main()
