from pathlib import Path

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

from src.common import SP500_PATH, APPLE_PATH, PRODUCTS_PATH, UNRATE_PATH
from src.stock_utils import load_all, normalize, derivative, smooth, read_csv

SMOOTH_SIZE = 10


def read_unrate(path: Path, start_date=None, end_date=None, normalize_data=True, sort_date=True):
    """ Read unemployment rate values from csv. Pick those between start and end """
    date, unrate = read_csv(path=path, start_date=start_date, end_date=end_date, sort_date=sort_date)
    return date, normalize(unrate) if normalize_data else unrate


def main():
    sp500_x, apple_x, \
    sp500_y, apple_y, \
    percent_sp500_y, percent_apple_y, \
    correct_apple_y, diff, items = load_all(SMOOTH_SIZE, PRODUCTS_PATH, SP500_PATH, APPLE_PATH)

    unrate_x, unrate_y = read_unrate(UNRATE_PATH)
    unrate_y = smooth(unrate_y, half_window=SMOOTH_SIZE)

    unrate_dx, unrate_dy = derivative(unrate_x, unrate_y)
    unrate_dy = smooth(unrate_dy, half_window=SMOOTH_SIZE)

    plt.figure()
    ax1 = plt.subplot(311)
    plt.plot(sp500_x, normalize(sp500_y), color='red', label='S&P500')
    plt.plot(apple_x, normalize(apple_y), color='royalblue', label='APPL')
    plt.axhline(0.0, color='black', alpha=0.3, zorder=-1)
    plt.legend()

    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(unrate_x, normalize(unrate_y), color='orange', label='Unemployment rate', marker='o', markersize=3)
    plt.axhline(0.0, color='black', alpha=0.3, zorder=-1)
    plt.legend()

    ax2 = plt.subplot(313, sharex=ax1)
    plt.plot(unrate_dx, unrate_dy, color='limegreen', label='Unemployment rate derivative', marker='o', markersize=3)
    plt.axhline(0.0, color='black', alpha=0.3, zorder=-1)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    register_matplotlib_converters()
    main()
