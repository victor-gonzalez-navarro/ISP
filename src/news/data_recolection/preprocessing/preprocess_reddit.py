import json
from datetime import datetime
from pathlib import Path

import nltk

FILE_PATH = Path(__file__).resolve().parents[0]
REDDIT_PATH = (FILE_PATH / '../../data/reddit_old').resolve()
OUT_PATH = (FILE_PATH / '../../data/preprocessed_reddit.json').resolve()

def init():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')

    # print('Using ', device_lib.list_local_devices()[-1].name)


def read_news(sort=True):
    news = []

    for file in REDDIT_PATH.iterdir():
        if file.is_file():
            with file.open('r') as f:
                data = json.load(f)['data']
            for article in data:
                news.append({
                    'src': 'reddit',
                    'url': article['full_link'],
                    'headline': article['title'],
                    'domain': article['domain'],
                    'date': datetime.utcfromtimestamp(article['created_utc']),
                })

    if sort:
        news = sorted(news, key=lambda x: x['date'])

    return news


def main():
    init()
    news = read_news(sort=True)

    with OUT_PATH.open('w') as f:
        json.dump(news, f, indent=4, sort_keys=True, default=str)


if __name__ == '__main__':
    main()
