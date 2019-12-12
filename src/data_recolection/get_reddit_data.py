from pathlib import Path
import urllib.request, json
from urllib.parse import quote
import time
import matplotlib.pyplot as plt
import datetime


def gen_query(words):
    if len(words) == 1:
        return quote(words[0])
    if len(words) > 0:
        return quote('|'.join(words))


QUERY_WORDS = [
    'Apple',
    'iPod', 'iPad', 'iPhone', 'MacOS', 'iTunes', 'Google', 'Microsoft', 'IBM', 'Samsung', 'Facebook', 'Siri', 'iCloud',
    'iMac', 'Intel', 'Safari', 'tvOS', ' MacBook'
]

OTHER = ['iPhone', 'iOS', 'App Store', 'MacOS', 'iTunes', 'Steve Wozniak', 'Google',
    'Microsoft', 'Siri', 'Steve Jobs', 'iCloud', 'IBM', 'iMAC', 'Intel', 'Samsung', 'Safari',
    'Facebook', 'tvOS', 'MacBook', 'Tim Cook'
]
QUERY = gen_query(QUERY_WORDS + [word.lower() for word in QUERY_WORDS[1:]])
FILE_PATH = Path(__file__).resolve().parents[0]
REDDIT_PATH = (FILE_PATH / '../../data/reddit').resolve()


class RedditAPI:
    URL = 'https://api.pushshift.io/reddit/submission/search/?'

    def __init__(self, requests_per_min=110):
        self.__requests_per_min = requests_per_min
        self.__wait_time = 60 / requests_per_min
        self.__next = time.time() + self.__wait_time

    def get(self, vars):
        wait = self.__next - time.time()
        if wait > 0:
            time.sleep(wait)
        self.__next = time.time() + self.__wait_time

        if vars:
            url = RedditAPI.URL
            for k, v in vars.items():
                url += k + '=' + str(v) + '&'
            url = url[:-1]
            # print(url)
            with urllib.request.urlopen(url) as url:
                return json.loads(url.read().decode())
        else:
            return {}


def timestamp(date: datetime.date):
    return int((date - datetime.date(year=1970, month=1, day=1)) / datetime.timedelta(seconds=1))


def download_all(start_year=2000, years=18):
    REDDIT_PATH.mkdir(parents=False, exist_ok=True)
    api = RedditAPI()

    date = datetime.date(year=start_year, month=1, day=1)

    for i in range(12 * years):
        if date.month == 12:
            next_date = datetime.date(year=date.year + 1, month=1, day=1)
        else:
            next_date = datetime.date(year=date.year, month=date.month + 1, day=1)

        print(date)
        data = api.get({
            'q': QUERY,
            'after': timestamp(date),
            'before': timestamp(next_date),
            'sort_type': 'score',
            'sort': 'desc',
            'subreddit': 'worldnews',
            'limit': 1000,
        })
        file_name = REDDIT_PATH / (date.strftime('%d-%m-%y') + '_' + next_date.strftime('%d-%m-%y') + '.json')
        with file_name.open('w') as f:
            f.write(json.dumps(data, indent=2, sort_keys=True))
        date = next_date


def main():
    # download_all()
    import numpy as np

    x = []
    y = []
    c = []
    for file in REDDIT_PATH.iterdir():
        if file.is_file():
            with file.open('r') as f:
                data = json.load(f)
                data = data["data"]
            for article in data:
                x.append(article['created_utc'])
                y.append(np.random.uniform())
                c.append('b')
            print(file.name, len(data))

    for file in (REDDIT_PATH / '..').iterdir():
        if file.is_file() and 'sp500' not in file.name:
            with file.open('r') as f:
                data = json.load(f)
            for article in data:
                x.append(timestamp(datetime.datetime.strptime(article['pub_date'], '%Y-%m-%dT%H:%M:%S+%f').date()))
                y.append(np.random.uniform())
                c.append('r')

    plt.scatter(x, y, c=c)
    plt.show()


if __name__ == '__main__':
    main()
