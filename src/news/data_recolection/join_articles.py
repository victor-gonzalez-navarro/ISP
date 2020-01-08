from pathlib import Path
from news.news_utils import read_news, sort_news, save_news

FILE_PATH = Path(__file__).resolve().parents[0]
WSJ_PATH = (FILE_PATH / '../../data/preprocessed_wsj6_.json').resolve()
NYT_PATH = (FILE_PATH / '../../data/preprocessed_content.json').resolve()
REDDIT_PATH = (FILE_PATH / '../../data/preprocessed_reddit.json').resolve()
OUT_PATH = (FILE_PATH / '../../data/preprocessed_all.json').resolve()


def main():
    news_wsj = read_news(WSJ_PATH, date_attribute='date', date_format='%Y-%m-%d %H:%M:%S', sort_date=False)
    news_nyt = read_news(NYT_PATH, date_attribute='date', date_format='%Y-%m-%d %H:%M:%S', sort_date=False)
    news_reddit = read_news(REDDIT_PATH, date_attribute='date', date_format='%Y-%m-%d %H:%M:%S', sort_date=False)

    # database: nyt/reddit/wsj
    # date
    # source: some of these newspapers or databases are in reality news aggregator while also maintaining their own
    #   articles. Source indicates the origin of the article, and it may not be the same as the database name
    # header: article title
    # important: [true/false/null] some databases may mark some articles as important
    # word_vector: list of parsed and preprocessed words from the body of the article
    # word_indexes: index of the words in word_vector for the embedded layer using Gensim
    # url

    news_all = []
    print('WSJ:', len(news_wsj), 'articles')
    for art in news_wsj:
        news_all.append({
            'database': 'wsj',
            'date': art['date'],
            'source': art['source'],
            'headline': art['title'],
            'url': art['url'],
            'important': art['important'],
            'word_indexes': art['word_indexes_big'],
            'word_vector': art['word_vector'],
            'word_indexes_title': art['word_indexes_big'],  # The headline was not parsed
            'word_vector_title': art['word_vector'],
            'content': art['content'],
            'summary': art['desc']
        })

    print('NYT:', len(news_nyt), 'articles')
    for art in news_nyt:
        news_all.append({
            'database': 'nyt',
            'date': art['date'],
            'source': art['src'],  # Same as 'nyt'
            'headline': art['txt'],  # The headline in nyt is very confusing and not trivial. Thus we are using the abstract
                                     # Which is the same as the summary in this case
            'url': art['url'],
            'important': None,
            'word_indexes': art['word_indexes_big'],
            'word_vector': art['word_vector'],
            'word_indexes_title': art['word_indexes'],
            'word_vector_title': art['word_list'],
            'content': art['content'],
            'summary': art['txt']
        })

    print('REDDIT:', len(news_reddit), 'articles')
    for art in news_reddit:
        news_all.append({
            'database': 'reddit',
            'date': art['date'],
            'source': art['domain'],
            'headline': art['headline'],
            'url': art['url'],
            'important': None,
            'word_indexes': None,
            'word_vector': None,
            'word_indexes_title': None,
            'word_vector_title': None,
            'content': None,
            'summary': None
        })

    print('Total', len(news_all), 'articles')
    save_news(OUT_PATH, sort_news(news_all))


if __name__ == '__main__':
    main()
