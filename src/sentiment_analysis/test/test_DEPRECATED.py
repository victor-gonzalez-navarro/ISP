from pathlib import Path
from datetime import datetime
import numpy as np

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

from src.sentiment_analysis.ground_truth import read_stocks

FILE_PATH = Path(__file__).resolve().parents[0]
IN_PATH = (FILE_PATH / '../../../data/preprocessed.json').resolve()
STOCK_PATH = (FILE_PATH / '../../../data/sp500.csv').resolve()
PLOT = False


class Item:
    def __init__(self, day, month, year, products):
        self.date = datetime(day=day, year=year, month=month)
        self.products = products


ITEMS = [
    Item(day=16, month=2,  year=2000, products=['PowerBook']),
    Item(day=19, month=7,  year=2000, products=['Power Mac', 'Cinema Display', 'Apple Pro Speakers']),
    Item(day=13, month=9,  year=2000, products=['iBook']),
    Item(day=9,  month=1,  year=2001, products=['PowerBook', 'Power Mac', 'Apple Pro Speakers']),
    Item(day=1,  month=5,  year=2001, products=['iBook']),
    Item(day=18, month=7,  year=2001, products=['Power Mac']),
    Item(day=8,  month=9,  year=2001, products=['Macintosh Server']),
    Item(day=23, month=10, year=2001, products=['iPod']),
    Item(day=7,  month=1,  year=2002, products=['iMac', 'iBook']),
    Item(day=29, month=4,  year=2002, products=['eMac']),
    Item(day=14, month=5,  year=2002, products=['Xserve']),
    Item(day=17, month=7,  year=2002, products=['iMac', 'iPod']),
    Item(day=13, month=8,  year=2002, products=['Power Macintosh']),
    Item(day=27, month=8,  year=2002, products=['Macintosh Server']),
    Item(day=7,  month=1,  year=2003, products=['PowerBook']),
    Item(day=10, month=2,  year=2003, products=['Xserve']),
    Item(day=28, month=4,  year=2003, products=['iPod']),
    Item(day=23, month=6,  year=2003, products=['Power Macintosh']),
    Item(day=16, month=9,  year=2003, products=['PowerBook']),
    Item(day=22, month=10, year=2003, products=['iBook']),
    Item(day=18, month=11, year=2003, products=['iMac']),
    Item(day=6,  month=1,  year=2004, products=['Xserve', 'iPod']),
    Item(day=8,  month=1,  year=2004, products=['iPod']),
    Item(day=7,  month=6,  year=2004, products=['AirPort']),
    Item(day=9,  month=6,  year=2004, products=['Power Macintosh']),
    Item(day=28, month=6,  year=2004, products=['Cinema Display']),
    Item(day=19, month=7,  year=2004, products=['iPod']),
    Item(day=31, month=8,  year=2004, products=['iMac']),
    Item(day=26, month=9,  year=2004, products=['iPod']),
    Item(day=11, month=1,  year=2005, products=['Mac Mini', 'iPod']),
    Item(day=23, month=2,  year=2005, products=['iPod']),
    Item(day=7,  month=9,  year=2005, products=['iPod']),
    Item(day=2,  month=8,  year=2005, products=['Apple Mouse']),
    Item(day=12, month=10, year=2005, products=['iPod']),
    Item(day=19, month=10, year=2005, products=['Power Macintosh']),
    Item(day=10, month=1,  year=2006, products=['iMac', 'iPod']),
    Item(day=14, month=2,  year=2006, products=['MacBook']),
    Item(day=28, month=2,  year=2006, products=['Mac Mini', 'IiPod']),
    Item(day=24, month=4,  year=2006, products=['MacBook']),
    Item(day=16, month=5,  year=2006, products=['MacBook']),
    Item(day=13, month=7,  year=2006, products=['iPod']),
    Item(day=7,  month=8,  year=2006, products=['Mac Pro', 'Xserve']),
    Item(day=6,  month=9,  year=2006, products=['iMac']),
    Item(day=12, month=9,  year=2006, products=['iPod']),
    Item(day=25, month=9,  year=2006, products=['iPod']),
    Item(day=21, month=3,  year=2007, products=['Apple TV']),
    Item(day=29, month=6,  year=2007, products=['iPhone']),
    Item(day=7,  month=8,  year=2007, products=['iMac', 'Apple Mouse', 'Apple Keyboard', 'Mac Mini']),
    Item(day=5,  month=9,  year=2007, products=['iPod']),
    Item(day=8,  month=1,  year=2008, products=['Xserve', 'Mac Pro']),
    Item(day=15, month=1,  year=2008, products=['MacBook', 'USB SuperDrive']),
    Item(day=5,  month=2,  year=2008, products=['iPhone']),
    Item(day=26, month=2,  year=2008, products=['MacBook']),
    Item(day=27, month=2,  year=2008, products=['iPod']),
    Item(day=29, month=2,  year=2008, products=['Time Capsule']),
    Item(day=17, month=3,  year=2008, products=['AirPort']),
    Item(day=28, month=4,  year=2008, products=['iMac']),
    Item(day=11, month=7,  year=2008, products=['iPhone']),
    Item(day=9,  month=9,  year=2008, products=['iPod']),
    Item(day=14, month=10, year=2008, products=['MacBook', 'LED Cinema Display']),
    Item(day=6,  month=1,  year=2009, products=['MacBook']),
    Item(day=29, month=1,  year=2009, products=['MacBook']),
    Item(day=3,  month=3,  year=2009, products=['Mac Mini', 'iMac', 'Mac Pro', 'Time Capsule', 'AirPort', 'Apple Keyboard']),
    Item(day=11, month=3,  year=2009, products=['iPod']),
    Item(day=7,  month=4,  year=2009, products=['Xserve']),
    Item(day=27, month=5,  year=2009, products=['MacBook']),
    Item(day=8,  month=6,  year=2009, products=['MacBook']),
    Item(day=19, month=6,  year=2009, products=['iPhone']),
    Item(day=30, month=7,  year=2009, products=['Time Capsule']),
    Item(day=9,  month=9,  year=2009, products=['iPod']),
    Item(day=20, month=10, year=2009, products=['iMac', 'MacBook', 'Mac Mini', 'Magic Mouse', 'AirPort']),
    Item(day=31, month=3,  year=2010, products=['Time Capsule']),
    Item(day=3,  month=4,  year=2010, products=['iPad']),
    Item(day=13, month=4,  year=2010, products=['MacBook']),
    Item(day=30, month=4,  year=2010, products=['iPad']),
    Item(day=18, month=5,  year=2010, products=['MacBook']),
    Item(day=15, month=6,  year=2010, products=['Mac Mini']),
    Item(day=24, month=6,  year=2010, products=['iPhone']),
    Item(day=27, month=7,  year=2010, products=['iMac', 'Magic Trackpad', 'Apple Battery Charger']),
    Item(day=9,  month=8,  year=2010, products=['Mac Pro']),
    Item(day=1,  month=9,  year=2010, products=['iPod', 'Apple TV']),
    Item(day=20, month=10, year=2010, products=['MacBook Air']),
    Item(day=10, month=2,  year=2011, products=['iPhone']),
    Item(day=24, month=2,  year=2011, products=['MacBook']),
    Item(day=11, month=3,  year=2011, products=['iPad']),
    Item(day=3,  month=5,  year=2011, products=['iMac']),
    Item(day=21, month=6,  year=2011, products=['AirPort', 'Time Capsule']),
    Item(day=20, month=7,  year=2011, products=['MacBook', 'Mac Mini', 'Thunderbolt Display']),
    Item(day=14, month=10, year=2011, products=['iPhone']),
    Item(day=24, month=10, year=2011, products=['MacBook']),
    Item(day=16, month=3,  year=2012, products=['iPad', 'Apple TV']),
    Item(day=11, month=6,  year=2012, products=['Mac Pro', 'MacBook', 'AirPort']),
    Item(day=12, month=9,  year=2012, products=['iPhone', 'iPod']),
    Item(day=21, month=9,  year=2012, products=['iPhone']),
    Item(day=11, month=10, year=2012, products=['iPod']),
    Item(day=23, month=10, year=2012, products=['Mac Mini', 'MacBook']),
    Item(day=2,  month=11, year=2012, products=['iPad']),
    Item(day=16, month=11, year=2012, products=['iPad']),
    Item(day=30, month=11, year=2012, products=['iMac']),
    # Item(day=??, month=12, year=2012, products=['']),
    Item(day=28, month=1,  year=2013, products=['Apple TV']),
    Item(day=13, month=2,  year=2013, products=['MacBook']),
    Item(day=30, month=5,  year=2013, products=['iPod']),
    Item(day=10, month=6,  year=2013, products=['AirPort', 'Time Capsule', 'MacBook']),
    Item(day=20, month=9,  year=2013, products=['iPhone']),
    Item(day=24, month=9,  year=2013, products=['iMac']),
    Item(day=22, month=10, year=2013, products=['MacBook']),
    Item(day=1,  month=11, year=2013, products=['iPad']),
    Item(day=12, month=11, year=2013, products=['iPad']),
    Item(day=19, month=12, year=2013, products=['Mac Pro']),
    Item(day=18, month=3,  year=2014, products=['iPhone']),
    Item(day=29, month=4,  year=2014, products=['MacBook']),
    Item(day=18, month=6,  year=2014, products=['iMac']),
    Item(day=26, month=6,  year=2014, products=['iPod']),
    Item(day=29, month=7,  year=2014, products=['MacBook']),
    Item(day=19, month=9,  year=2014, products=['iPhone']),
    Item(day=16, month=10, year=2014, products=['iMac', 'Mac Mini']),
    Item(day=22, month=10, year=2014, products=['iPad']),
    Item(day=9,  month=3,  year=2015, products=['MacBook']),
    Item(day=10, month=4,  year=2015, products=['MacBook']),
    Item(day=24, month=4,  year=2015, products=['Apple Watch']),
    Item(day=19, month=5,  year=2015, products=['MacBook', 'iMac']),
    Item(day=15, month=7,  year=2015, products=['iPod']),
    Item(day=9,  month=9,  year=2015, products=['iPad']),
    Item(day=25, month=9,  year=2015, products=['iPhone']),
    Item(day=13, month=10, year=2015, products=['iMac', 'Magic Mouse', 'Magic Trackpad', 'Magic Keyboard']),
    Item(day=30, month=10, year=2015, products=['Apple TV']),
    Item(day=11, month=11, year=2015, products=['iPad', 'Apple Pencil']),
    Item(day=31, month=3,  year=2016, products=['iPad', 'iPhone']),
    Item(day=19, month=4,  year=2016, products=['MacBook']),
    Item(day=7,  month=9,  year=2016, products=['iPad']),
    Item(day=16, month=9,  year=2016, products=['iPhone', 'Apple Watch']),
    Item(day=27, month=10, year=2016, products=['MacBook Pro']),
    Item(day=28, month=10, year=2016, products=['Apple Watch']),
    Item(day=19, month=12, year=2016, products=['AirPods']),
    Item(day=21, month=3,  year=2017, products=['iPhone']),
    Item(day=24, month=3,  year=2017, products=['iPad']),
    Item(day=5,  month=6,  year=2017, products=['iPad', 'MacBook', 'iMac', 'Magic Keyboard']),
    Item(day=22, month=9,  year=2017, products=['Apple TV', 'Apple Watch', 'iPhone']),
    Item(day=3,  month=11, year=2017, products=['iPhone']),
    Item(day=14, month=12, year=2017, products=['iMac']),
    Item(day=9,  month=2,  year=2018, products=['HomePod']),
    Item(day=27, month=3,  year=2018, products=['iPad']),
    Item(day=12, month=7,  year=2018, products=['MacBook']),
    Item(day=21, month=9,  year=2018, products=['Apple Watch', 'iPhone']),
    Item(day=26, month=10, year=2018, products=['iPhone']),
    Item(day=30, month=10, year=2018, products=['Apple Pencil']),
    Item(day=7,  month=11, year=2018, products=['iPad', 'MacBook', 'Mac Mini']),
]


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
    return [(y[i] + 0.0001) / (y[i - 1] + 0.0001) for i in range(1, len(y))] + [1.0]


def main():
    # news = read_news(IN_PATH)

    plt.figure()
    sp_x, sp_y = read_stocks(STOCK_PATH, normalize=False)
    aapl_x, aapl_y = read_stocks((FILE_PATH / '../../../data/AAPL.csv').resolve(), normalize=False)
    sp_y = smooth(sp_y, half_window=10)
    aapl_y = smooth(aapl_y, half_window=10)

    start_aapl_y = normalize(aapl_y)[0]

    ax1 = plt.subplot(221)
    plt.plot(sp_x, normalize(sp_y), color='red')
    plt.plot(aapl_x, normalize(aapl_y), color='blue')
    plt.gca().set_title('basic')

    for item in ITEMS:
        if 'iPhone' in item.products:
            plt.axvline(x=item.date)

    sp_y = to_percent(sp_y)
    aapl_y = to_percent(aapl_y)
    sp_y = smooth(sp_y, half_window=10)
    aapl_y = smooth(aapl_y, half_window=10)

    ax2 = plt.subplot(222, sharex=ax1)
    plt.plot(sp_x, sp_y, color='red')
    plt.plot(aapl_x, aapl_y, color='blue')
    plt.axhline(1.0)
    plt.gca().set_title('percent')

    for item in ITEMS:
        if 'iPhone' in item.products:
            plt.axvline(x=item.date)

    ax3 = plt.subplot(224, sharex=ax1)
    plt.plot(sp_x, np.clip((1.0 - np.array(sp_y)) * (np.array(aapl_y) - 1.0), 0, 9999999), color='green')
    plt.axhline(0.0)
    plt.gca().set_title('percent')

    for item in ITEMS:
        if 'iPhone' in item.products:
            plt.axvline(x=item.date)

    y = [start_aapl_y]
    for i in range(len(aapl_y)):
        d = abs(sp_y[i] - 1) * 2
        print(d)
        b = y[-1] * aapl_y[i]
        if sp_y[i] > 1.0:
            y.append(b * (1 - d))
        else:
            y.append(b * (1 + d))


    ax4 = plt.subplot(223, sharex=ax1)
    plt.plot(aapl_x, y[:-1], color='green')
    plt.gca().set_title('percent')

    for item in ITEMS:
        if 'iPhone' in item.products:
            plt.axvline(x=item.date)

    # begin = datetime(day=1, month=1, year=2014)
    # end = datetime(day=1, month=1, year=2015)

    # y = [j for i, j in enumerate(y) if begin < x[i] < end]
    # x = [i for i in x if begin < i < end]
    # items = [item for item in ITEMS if begin < item.date < end]
    # news = [new for new in news if datetime(year=2014, month=9, day=1) < new['date'] < datetime(year=2014, month=11, day=30)]

    # for new in news:
    #     print(new['txt'])

    # cy = (max(y) - min(y)) / 2.0 + min(y)

    # plt.figure()
    # plt.plot(x, y, alpha=0.8, label='S&P Stock', zorder=0)
    # plt.plot(x2, y2, alpha=0.8, label='AAPL Stock', zorder=0)
    # plt.plot(x2, np.array(y) - np.array(y2), alpha=0.8, label='Diff Stock', zorder=0)
    # plt.xticks(rotation=30)
    # plt.legend()

    # for item in items:
    #     plt.axvline(x=item.date, alpha=0.5, color='orange')
    #     plt.text(item.date, cy, ' '.join(item.products), rotation=90)

    plt.show()


if __name__ == '__main__':
    register_matplotlib_converters()
    main()
