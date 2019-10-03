from NYTApi import NYTApi
from pathlib import Path
import json
import time


QUERY = 'apple'
BEGIN_DATE = '20120101'
END_DATE = '20120101'

FILE_PATH = Path(__file__).resolve().parents[0]
TOKEN_PATH = (FILE_PATH / '../../config/token').resolve()
DATA_PATH = (FILE_PATH / '../../data').resolve()


def gen_identifier():
    return str(int(time.time() / 1000))


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
    results = api.search(QUERY, BEGIN_DATE, END_DATE)

    print('Saving json...')
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    file_path = DATA_PATH / (QUERY + '_' + BEGIN_DATE + '_' + END_DATE + '_' + gen_identifier() + '.json')
    with file_path.open('w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
