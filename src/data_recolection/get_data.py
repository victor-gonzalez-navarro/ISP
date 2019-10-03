from NYTApi import NYTApi
from pathlib import Path


FILE_PATH = Path(__file__).resolve().parents[0]
TOKEN_PATH = (FILE_PATH / '../../config/token').resolve()


def main():
    try:
        with TOKEN_PATH.open('r') as f:
            token = f.read()
    except FileNotFoundError:
        print('File %s not found' % TOKEN_PATH)
        return
    except IOError:
        print('File %s error' % TOKEN_PATH)
        return

    api = NYTApi(token)
    result = api.search('apple', begin_date='20120101', end_date='20120101')

    for article in result['response']['docs']:
        print(article['abstract'])


if __name__ == '__main__':
    main()
