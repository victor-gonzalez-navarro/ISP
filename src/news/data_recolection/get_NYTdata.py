from news.data_recolection.NYTApi import NYTApi
from pathlib import Path
from datetime import date
import json
import time


QUERY = 'apple'
BEGIN_DATE = date(year=2017, month=1, day=1).strftime("%Y%m%d")
END_DATE = date(year=2017, month=12, day=31).strftime("%Y%m%d")

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

    for year in range(2016, 2020):
        begin_date = date(year, month=1, day=1).strftime("%Y%m%d")
        end_date = date(year, month=12, day=31).strftime("%Y%m%d")
        results = api.search(QUERY, begin_date, end_date)
        print('Saving json...')
        DATA_PATH.mkdir(parents=True, exist_ok=True)
        file_path = DATA_PATH / (QUERY + '_' + begin_date + '_' + end_date + '_' + gen_identifier() + '.json')
        with file_path.open('w') as f:
            json.dump(results, f)


if __name__ == '__main__':
    main()
