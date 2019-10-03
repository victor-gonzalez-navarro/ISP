import urllib.request, json


class NYTApi:

    BASE_SEARCH = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
    ITEMS_PER_PAGE = 10

    def __init__(self, token):
        self.__token = token

    def search(self, query, begin_date=None, end_date=None):
        url = NYTApi.BASE_SEARCH + '?'
        url += 'q=' + query

        if begin_date:
            url += '&begin_date=' + begin_date
        if end_date:
            url += '&end_date=' + end_date

        url += '&api-key=' + self.__token
        return self.__get_json(url)

    def __get_json(self, url):
        hits = int('+inf')
        page = 0

        while page * ITEMS_PER_PAGE < hits:
            with urllib.request.urlopen(url + '&page=%d' % page) as response:
                data = json.loads(response.read().decode())
                hits = data['response']['meta']['hits']

