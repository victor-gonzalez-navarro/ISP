import re
import nltk
from datetime import datetime
import json
from random import sample
from pathlib import Path
from nltk.corpus import wordnet
#from sentiment_analysis.w2v_models import GensimModel
from utils import logger
#from tensorflow.python.client import device_lib
from tqdm import tqdm

QUERY = 'Apple'
FILE_PATH = Path(__file__).resolve().parents[0]
REDDIT_PATH = (FILE_PATH / '../../data/reddit').resolve()
NYT_PATH = (FILE_PATH / '../../data/nyt').resolve()
OUT_PATH = (FILE_PATH / '../../data/preprocessed.json').resolve()
CAPITAL = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
DATASETS = ['nyt']  # nyt, reddit


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

    #print('Using ', device_lib.list_local_devices()[-1].name)


def read_news(datasets, sort=True, remove_duplicated=True):

    if remove_duplicated and not sort:
        print('WARNING, remove_duplicated requires sorting. sort set to True')
        sort = True

    news = []
    if 'reddit' in datasets:
        logger.d('Not implemented')

    if 'nyt' in datasets:
        query = 'Apple'  # TODO
        for file in NYT_PATH.iterdir():
            if file.is_file():
                with file.open('r') as f:
                    data = json.load(f)
                for article in data:
                    txt = article['abstract'] or article['snippet'] or article['lead_paragraph']
                    if txt and (article['document_type'] == 'article') and (not query or query in txt):
                        news.append({
                            'src': 'nyt',
                            'url': article['web_url'],
                            'txt': txt,
                            'date': datetime.strptime(article['pub_date'], '%Y-%m-%dT%H:%M:%S+%f'),
                        })

    if sort:
        news = sorted(news, key=lambda x: x['date'])

        if remove_duplicated:
            unique_news = []
            for i in range(len(news) - 1):
                if news[i]['txt'] != news[i + 1]['txt']:
                    unique_news.append(news[i])
            news = unique_news

    return news


def contains_any(text, text_list):
    for text_token in text_list:
        if text_token in text:
            return True
    return False


def is_preposition_uppercase_a(tokens, idx):
    if tokens[idx] != 'A':
        return False

    after_symbol = (idx == 0 or not tokens[idx - 1].isalpha())

    prev_mayus = (idx > 0) and (tokens[idx - 1][0] in CAPITAL)
    prev_mayus = (idx > 1) and (prev_mayus or (tokens[idx - 2][0] in CAPITAL))
    next_single = (idx < (len(tokens) - 1)) and len(tokens[idx + 1]) == 1

    return after_symbol or (prev_mayus and not next_single)


def is_subject_uppercase_i(tokens, idx):
    return tokens[idx] == 'I'


def any_case(text):
    text = text.lower()
    return '({:s}|{:s}|{:s})'.format(text.upper(), text[0].upper() + text[1:], text)


def tokens_span(txt):
    spans = []
    tokens = nltk.word_tokenize(txt)
    offset = 0
    for token in tokens:
        offset = txt.find(token, offset)
        spans.append((offset, offset+len(token)))
        offset += len(token)

    return tokens, spans


def preprocess(news, encoding='UTF-8'):
    items = 'photo|map|drawing|chart|diagram|recipe|list|graph|logo|illustration|' \
            '(other photos)|cartoon|(photo, drawing)|(diagram of balcony)|(photo \\(Pulse column\\))'
    
    regex_tags = re.compile(r'[;,]\s*(%s)s?([;,]\s(%s)s?)*\s*\(.{1,3}\).{0,30}' % (items, items))
    regex_spaces = re.compile(r'[\s\r\n]+')
    regex_size = re.compile(r'\(.\)$')
    regex_author = re.compile(r'\b[A-Z]\s+[A-Z]\s+[A-Z][A-Za-z]{1,10}(\s+Jr)?\b')
    regex_rwapple = re.compile(r'\bR\s?\.?(\s+)?W\s?\.?\s+Apple(\'s|[,.;])?\s+(jr\.?)?', re.IGNORECASE)

    # Remove headlines that may not refer to the company
    news = [article for article in news if not re.findall(r'\b(applefarm|apples|fruits?|trees?|((golden|red|green)\s+apples?)|(apples?\s+pies?))\b', article['txt'], re.IGNORECASE)]

    # Remove specific lines
    TO_REMOVE = ['GORDON-Edward. To our beloved Dad and Papa', 'MUSCHEL-Chaim. Young Israel', '6449 Letzing', 'Big Apple Circus', 'Selborne Journal: Monday, 24 May 1784', 'A selective listing by critics of The Times: New or noteworthy Broadway and Off Broadway shows this weekend', 'EMBLEMS Animal', 'EMBLEMS Bird', 'RARELY can store-bought fruit match']
    news = [article for article in news if not contains_any(article['txt'], TO_REMOVE)]

    # Remove all headlines by R W Apple
    news = [article for article in news if not regex_rwapple.search(article['txt'])]

    # Basic pre-processing
    for i, article in enumerate(news):
        headline = article['txt']
        headline = headline.encode(encoding, errors='strict').decode(encoding)  # Assert no encoding errors
        headline = headline.replace('’', '\'')
        headline = headline.replace('‘', '\'')
        headline = regex_spaces.sub(' ', headline)  # Remove multiple spaces
        headline = regex_tags.sub('', headline)  # Remove tags ; photos; drawings (M) ...
        headline = regex_size.sub('', headline)  # Remove size indicator (M), (L), ...
        headline = regex_spaces.sub(' ', headline)  # Repeat this operation just in case
        headline = regex_author.sub('Author', headline)
        headline = headline.replace('Q.', '')  # Remove Q. (Question) Indicator
        headline = headline.replace('Q\'s', '')  # Remove Q. (Question) Indicator
        news[i]['txt'] = headline.strip()  # Trim string

    # Decontraction
    for i, article in enumerate(news):
        headline = article['txt']
        headline = re.sub(r"won\'t", "will not", headline)
        headline = re.sub(r"can\'t", "can not", headline)
        headline = re.sub(r"n\'t", " not", headline)
        headline = re.sub(r"\'re", " are", headline)
        headline = re.sub(r"\'s", " is", headline)
        headline = re.sub(r"\'d", " would", headline)
        headline = re.sub(r"\'ll", " will", headline)
        headline = re.sub(r"\'t", " not", headline)
        headline = re.sub(r"\'ve", " have", headline)
        headline = re.sub(r"\'m", " am", headline)
        news[i]['txt'] = headline

    # Check if any tags or size was not removed
    for article in news:
        headline = article['txt']
        if re.search('; .{1,20} \(.\)', headline) or re.search('\(.\)$', headline):
            logger.e('Headline may not be processed correctly:', headline[-100:])

    # Check for names
    regex_name = re.compile(r'([A-Z][a-z]+)\s+[A-Z]\.?\s+([A-Z][a-z]+)')
    regex_osurname = re.compile(r'O\'([A-Z][a-z]+)')
    for i, _ in enumerate(news):
        news[i]['txt'] = regex_osurname.sub('\\1', news[i]['txt'])
        news[i]['txt'] = regex_name.sub('\\1 \\2', news[i]['txt'])
        news[i]['txt'] = news[i]['txt'].replace('Steven', 'Steve')  # Steve and Steven Jobs

    # iPhone names
    regex_iphone = re.compile(r'i[Pp]hone\s+(\d\d|\dG?S?c?s?|SE|XR?S?)(\s+S)?(\s+(Plus|Max|Pro))?(\s+(Plus|Max|Pro))?')
    for i, article in enumerate(news):
        news[i]['txt'] = regex_iphone.sub('iphone', article['txt'])  # iPhone names

    # Remove some specific contraptions from some companies
    for i, _ in enumerate(news):
        if news[i]['txt'].endswith('. S'):
            news[i]['txt'] = news[i]['txt'].replace('. S', '')
        news[i]['txt'] = news[i]['txt'].replace('À', 'A')
        news[i]['txt'] = news[i]['txt'].replace('A. A. C.', 'AAC')
        news[i]['txt'] = news[i]['txt'].replace('Toys \'R\' Us', 'ToysRUs')
        news[i]['txt'] = news[i]['txt'].replace('&#038;', '&')
        news[i]['txt'] = news[i]['txt'].replace('AT&T', 'ATT')
        news[i]['txt'] = news[i]['txt'].replace('S.&P.', 'SP')
        news[i]['txt'] = news[i]['txt'].replace('S&P.', 'SP')
        news[i]['txt'] = news[i]['txt'].replace('S&P', 'SP')
        news[i]['txt'] = news[i]['txt'].replace('R&B', 'RB')
        news[i]['txt'] = news[i]['txt'].replace('I&S', 'IS')
        news[i]['txt'] = news[i]['txt'].replace('J.P. Morgan', 'JP Morgan')
        news[i]['txt'] = news[i]['txt'].replace('J. P. Morgan', 'JP Morgan')
        news[i]['txt'] = news[i]['txt'].replace('J. P Morgan', 'JP Morgan')
        news[i]['txt'] = news[i]['txt'].replace('F.B.I.', 'FBI')
        news[i]['txt'] = news[i]['txt'].replace('hip-hop', 'hiphop')
        news[i]['txt'] = news[i]['txt'].replace('M&A', 'MA')
        news[i]['txt'] = news[i]['txt'].replace('M&M', 'MM')
        news[i]['txt'] = news[i]['txt'].replace('OS X', 'OSX')
        news[i]['txt'] = news[i]['txt'].replace('Verizon is V Cast service', 'Verizon')
        news[i]['txt'] = news[i]['txt'].replace('Galaxy S II', 'GalaxyS')
        news[i]['txt'] = news[i]['txt'].replace('Galaxy S III', 'GalaxyS')
        news[i]['txt'] = news[i]['txt'].replace('S II', 'GalaxyS')
        news[i]['txt'] = news[i]['txt'].replace('S III', 'GalaxyS')
        news[i]['txt'] = news[i]['txt'].replace('Nexus Q', 'NexusQ')
        news[i]['txt'] = news[i]['txt'].replace('J.& J.', 'JJ')
        news[i]['txt'] = news[i]['txt'].replace('Mac OX X', 'Mac')
        news[i]['txt'] = news[i]['txt'].replace('iTunes U', 'iTunes')
        news[i]['txt'] = news[i]['txt'].replace('Zen Vision:M', 'Zen Vision')
        news[i]['txt'] = news[i]['txt'].replace('Vision:M', 'Vision')
        news[i]['txt'] = news[i]['txt'].replace('Apple I', 'Apple Computer')
        news[i]['txt'] = news[i]['txt'].replace('Final Cut Pro X', 'VideoEditor')
        news[i]['txt'] = news[i]['txt'].replace('The Elder Scrolls V: Skyrim', 'Game')

        news[i]['txt'] = news[i]['txt'].replace('C.E.O.', 'CEO')
        news[i]['txt'] = news[i]['txt'].replace('iMac', 'Mac')
        news[i]['txt'] = news[i]['txt'].replace('Apple ComputerIE', 'Apple')
        news[i]['txt'] = news[i]['txt'].replace('Apple-1', 'Apple')
        news[i]['txt'] = news[i]['txt'].replace('Apple Computernc', 'Apple')
        news[i]['txt'] = news[i]['txt'].replace('Apple Computer', 'Apple')
        news[i]['txt'] = news[i]['txt'].replace('Apple Co', 'Apple')
        news[i]['txt'] = news[i]['txt'].replace('Apple Co.', 'Apple')
        news[i]['txt'] = news[i]['txt'].replace('Apple Company', 'Apple')
        news[i]['txt'] = news[i]['txt'].replace('Apple Corps Ltd.', 'Apple')
        news[i]['txt'] = news[i]['txt'].replace('Apple Corps', 'Apple')
        news[i]['txt'] = news[i]['txt'].replace('Apple Inc', 'Apple')
        news[i]['txt'] = news[i]['txt'].replace('Apple Inc.', 'Apple')
        news[i]['txt'] = news[i]['txt'].replace('Apple computers', 'Apple')
        news[i]['txt'] = news[i]['txt'].replace('Apple computer', 'Apple')
        news[i]['txt'] = news[i]['txt'].replace('Apple computers.', 'Apple')
        news[i]['txt'] = news[i]['txt'].replace('Apple computer.', 'Apple')

    # Remove single words in mayus
    for k, article in enumerate(news):
        headline = article['txt']
        tokens, spans = tokens_span(headline)

        offset = 0
        for i, token in enumerate(tokens):
            if len(token) == 1 and token.isupper() and token.isalnum() and not is_subject_uppercase_i(tokens, i) and not is_preposition_uppercase_a(tokens, i):
                news[k]['txt'] = news[k]['txt'][:spans[i][0] + offset] + news[k]['txt'][spans[i][1] + offset:]
                offset -= 1

        news[k]['txt'] = regex_spaces.sub(' ', news[k]['txt'])  # Remove multiple spaces

    # Check for words that are all in mayus
    #unknown = set()
    #for headline in headlines[:100]:
        #    tokens = nltk.tokenize.word_tokenize(headline)
        #a = False
        #for token in tokens:
        #    if token.isalnum() and token.isupper():
        #        unknown.add(token)
        #        print(token, '---'.join(i.definition() for i in wordnet.synsets(token)))
        #        a = True
        #if a:
        #    print(headline)
    #    print()

    # How many w2v
    if False:
        print('Loading model...')
        word2vec_model = GensimModel('glove-wiki-gigaword-100')  # BOWModel() # 2513 wiki
        print('Done')
        unknown = set()
        for article in tqdm(news):
            tokens = nltk.tokenize.word_tokenize(article['txt'])
            for token in tokens:
                try:
                    word2vec_model.word2index(token)
                except KeyError:
                    try:
                        word2vec_model.word2index(token.lower())
                    except KeyError:
                        unknown.add(token)
                    # print(token)

        for word in sorted(list(unknown)):
            print(word)
        exit()

    # Remove non words
    return news


def main():
    init()

    # stop_words = set(nltk.corpus.stopwords.words('english'))

    logger.i('Reading news...')
    news = read_news(DATASETS, sort=True, remove_duplicated=True)

    logger.i('A total of {:d} news:'.format(len(news)))
    for i, article in enumerate(sample(news, k=2)):
        logger.i('[{:d}]  {:s}'.format(i, article['txt']))

    before = len(news)
    news = preprocess(news)
    after = len(news)
    print('Before', before, 'After', after, 'Removed', before - after)

    with OUT_PATH.open('w') as f:
        json.dump(news, f, indent=4, sort_keys=True, default=str)


if __name__ == '__main__':
    main()
