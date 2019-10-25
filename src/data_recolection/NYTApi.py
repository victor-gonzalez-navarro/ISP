import json
import time
import math
import urllib.request
from pathlib import Path
from datetime import date, datetime

FILE_PATH = Path(__file__).resolve().parents[0]
API_LIMITS_PATH = (FILE_PATH / '../../config/nyt_api_limits.json').resolve()


class NYTApi:

    BASE_SEARCH = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
    ITEMS_PER_PAGE = 10

    def __init__(self, token):
        self.__token = token
        self.__last_query = float('-inf')
        self.__limits = {}
        self.__read_limits()

    def __del__(self):
        self.__save_limits()

    def __save_limits(self):
        try:
            API_LIMITS_PATH.parents[0].mkdir(parents=True, exist_ok=True)
            with API_LIMITS_PATH.open('w') as f:
                json.dump(self.__limits, f)
        except IOError:
            print('NYTAPI Error: Could not save api limits')

    def __read_limits(self):
        self.__limits = {}
        try:
            with API_LIMITS_PATH.open('r') as f:
                self.__limits = json.loads(f.read())
        except IOError:
            print('NYTAPI Error: Could not read api limits')

        if 'date' not in self.__limits:
            self.__limits['date'] = date.today().strftime("%d-%m-%Y")
        if 'queries' not in self.__limits:
            self.__limits['queries'] = 0
        if 'limit_day' not in self.__limits:
            self.__limits['limit_day'] = 4000
        if 'limit_min' not in self.__limits:
            self.__limits['limit_min'] = 10

        try:
            last_date = datetime.strptime(self.__limits['date'], '%d-%m-%Y')
        except ValueError:
            print('NYTAPI Error: Invalid date in api limit file')
            last_date = date.today()

        # If different day, reset queries
        now = date.today()
        if last_date.day != now.day or last_date.month != now.month or last_date.year != now.year:
            self.__limits['queries'] = 0

        # Set today's date
        self.__limits['date'] = date.today().strftime("%d-%m-%Y")

    def search(self, query, begin_date=None, end_date=None):
        url = NYTApi.BASE_SEARCH + '?'
        url += 'q=' + query

        if begin_date:
            url += '&begin_date=' + begin_date
        if end_date:
            url += '&end_date=' + end_date

        url += '&api-key=' + self.__token
        data = self.__get_all_json(url)
        self.__save_limits()
        return data

    def __get_all_json(self, url):
        hits = float('+inf')
        page = 0
        results = []

        while page * NYTApi.ITEMS_PER_PAGE < hits:
            try:
                data = self.__get_json(url + '&page=%d' % page)
                hits = data['response']['meta']['hits']
                results += data['response']['docs']
            except Exception as e:
                pass
            pages = int(math.ceil(hits / NYTApi.ITEMS_PER_PAGE))
            eta = (pages - page) * 60 / self.__limits['limit_min']
            page += 1
            print('\rPage %d of %d (ETA %ds)' % (page, pages, eta), end='')
        print()

        return results

    def __get_json(self, url):
        if self.__limits['queries'] >= self.__limits['limit_day']:
            raise Exception('Max daily queries reached')

        interval = 60 / self.__limits['limit_min']  # seconds
        elapsed = time.time() - self.__last_query
        wait = interval - elapsed

        if wait > 0:
            time.sleep(wait)

        self.__limits['queries'] += 1
        with urllib.request.urlopen(url) as response:
            res = json.loads(response.read().decode())

        self.__last_query = time.time()
        return res
