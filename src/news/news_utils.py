import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

from src.news.common import SP500_PATH, APPLE_PATH, PRODUCTS_PATH, NEWS_PATH, WINDOWS_SIZES
from src.news.stock_utils import normalize
from src.news.stock_utils import load_all

SMOOTH_SIZE = 10


def sort_news(news, date_attribute='date'):
    return sorted(news, key=lambda k: k[date_attribute])


def read_news_all(path: Path, date_attribute=None, date_format='%d-%m-%y', sort_date=True, databases=None):
    if databases is None:
        raise Exception('Please specify one or more databases to read the news')

    if 'reddit' in databases:
        print('WARNING: Reddit not implemented in this version')

    if isinstance(databases, str):
        databases = [databases]

    news = read_news(path, date_attribute, date_format, sort_date)
    news = [article for article in news if article['database'] in databases]
    return news


def read_news(path: Path, date_attribute=None, date_format='%d-%m-%y', sort_date=True):
    with path.open('r') as f:
        data = json.load(f)

    for item in data:
        item[date_attribute] = datetime.strptime(item[date_attribute], date_format)

    if sort_date:
        return sort_news(data, date_attribute)
    else:
        return data


def encode_articles(news, indent=2):
    txt = '[\n'
    from tqdm import tqdm
    for i, article in enumerate(tqdm(news)):
        txt += (' ' * indent) + '{\n'
        items = list(article.items())
        for k, v in items[:-1]:
            txt += '{:s}"{:s}": {:s},\n'.format(' ' * indent * 2, k, json.dumps(v, default=str))
        k, v = items[-1]  # Last one without comma
        txt += '{:s}"{:s}": {:s}\n'.format(' ' * indent * 2, k, json.dumps(v, default=str))
        if i == (len(news) - 1):
            txt += (' ' * indent) + '}\n'
        else:
            txt += (' ' * indent) + '},\n'
    txt += ']'
    return txt


def save_news(path: Path, news, indent=2):
    with path.open('w') as f:
        f.write(encode_articles(news, indent))


def bins(news, delta, begin_date, end_date):
    news = sort_news(news)  # In case they are not sorted
    counts = []
    begins = []
    middles = []

    current = begin_date
    while current < end_date:
        next_date = current + delta

        try:
            count = next(i for i, v in enumerate(news) if v['date'] >= next_date)
        except StopIteration:
            count = 0

        news = news[count:]

        counts.append(count)
        begins.append(current)
        middles.append(current + (delta / 2))
        current = next_date

    return begins, middles, counts


def prune_news(news, max_date: datetime):
    """ Remove news that are posterior to max_date """
    return [article for article in news if article['date'] < max_date]


def add_labels(x, y, news, prev_window=0, next_windows=WINDOWS_SIZES, multiplier=1, limits=None):
    assert all(news[i]['date'] <= news[i + 1]['date'] for i in range(len(news) - 1))  # Assert sorted

    offset = 0
    for article in news:
        idx = next(i for i, v in enumerate(x[offset:]) if v >= article['date']) + offset
        start = max(0, idx - prev_window)

        article['windows'] = {}
        for next_window in next_windows:
            end = min(len(y) - 1, idx + next_window)
            value = (y[end] / y[start]) * multiplier

            if limits and (value - multiplier) > limits:
                article['windows'][next_window] = multiplier + limits
            elif limits and (multiplier - value) > limits:
                article['windows'][next_window] = multiplier - limits
            else:
                article['windows'][next_window] = value

        offset = idx

    return news


def main():
    sp500_x, apple_x, \
    sp500_y, apple_y, \
    percent_sp500_y, percent_apple_y, \
    correct_apple_y, diff, items = load_all(SMOOTH_SIZE, PRODUCTS_PATH, SP500_PATH, APPLE_PATH)

    databases = ['reddit', 'nyt', 'wsj']
    news = read_news_all(NEWS_PATH, date_attribute='date', date_format='%Y-%m-%d %H:%M:%S', sort_date=True, databases=databases)

    reddit_news = [article for article in news if article['database'] == 'reddit']
    nyt_news = [article for article in news if article['database'] == 'nyt']
    wsj_news = [article for article in news if article['database'] == 'wsj']

    delta = timedelta(days=100)
    _, reddit_dates, reddit_counts = bins(reddit_news, delta, begin_date=news[0]['date'], end_date=news[-1]['date'])
    _, nyt_dates, nyt_counts = bins(nyt_news, delta, begin_date=news[0]['date'], end_date=news[-1]['date'])
    _, wsj_dates, wsj_counts = bins(wsj_news, delta, begin_date=news[0]['date'], end_date=news[-1]['date'])

    fig, ax1 = plt.subplots()
    ax1.plot(reddit_dates, reddit_counts, color='limegreen', label='Reddit articles', marker='o', markersize=4)
    ax1.fill_between(reddit_dates, 0, reddit_counts, color='limegreen', alpha=0.2)
    ax1.plot(nyt_dates, nyt_counts, color='orange', label='NYT articles', marker='x', markersize=4)
    ax1.fill_between(nyt_dates, 0, nyt_counts, color='orange', alpha=0.2)
    ax1.plot(wsj_dates, wsj_counts, color='brown', label='WSJ articles', marker='D', markersize=4)
    ax1.fill_between(wsj_dates, 0, wsj_counts, color='brown', alpha=0.2)
    ax1.set_ylabel('Frequency')
    ax1.legend()

    ax2 = ax1.twinx()
    ax2.plot(sp500_x, normalize(sp500_y), color='red', label='S&P500 stock value', alpha=0.8)
    ax2.plot(apple_x, normalize(apple_y), color='royalblue', label='APPL stock value', alpha=0.8)
    ax2.set_ylabel('Normalized value')
    ax2.legend()

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    register_matplotlib_converters()
    main()
