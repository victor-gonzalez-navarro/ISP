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
    headlines = [article['txt'] for article in news]

    items = 'photo|map|drawing|chart|diagram|recipe|list|graph|logo|illustration|' \
            '(other photos)|cartoon|(photo, drawing)|(diagram of balcony)|(photo \\(Pulse column\\))'
    
    regex_tags = re.compile(r'[;,]\s*(%s)s?([;,]\s(%s)s?)*\s*\(.{1,3}\).{0,30}' % (items, items))
    regex_spaces = re.compile(r'[\s\r\n]+')
    regex_size = re.compile(r'\(.\)$')
    regex_author = re.compile(r'\b[A-Z]\s+[A-Z]\s+[A-Z][A-Za-z]{1,10}(\s+Jr)?\b')
    regex_rwapple = re.compile(r'\bR\s?\.?(\s+)?W\s?\.?\s+Apple(\'s|[,.;])?\s+(jr\.?)?', re.IGNORECASE)

    # Remove headlines that may not refer to the company
    headlines = [headline for headline in headlines if not re.findall(r'\b(applefarm|apples|fruits?|trees?|((golden|red|green)\s+apples?)|(apples?\s+pies?))\b', headline, re.IGNORECASE)]

    # Remove specific lines
    TO_REMOVE = ['GORDON-Edward. To our beloved Dad and Papa', 'MUSCHEL-Chaim. Young Israel', '6449 Letzing', 'Big Apple Circus', 'Selborne Journal: Monday, 24 May 1784', 'A selective listing by critics of The Times: New or noteworthy Broadway and Off Broadway shows this weekend', 'EMBLEMS Animal', 'EMBLEMS Bird', 'RARELY can store-bought fruit match']
    headlines = [headline for headline in headlines if not contains_any(headline, TO_REMOVE)]

    # Remove all headlines by R W Apple
    headlines = [headline for headline in headlines if not regex_rwapple.search(headline)]

    # Basic pre-processing
    for i, headline in enumerate(headlines):
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
        headlines[i] = headline.strip()  # Trim string

    # Decontraction
    for i, headline in enumerate(headlines):
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
        headlines[i] = headline

    # Check if any tags or size was not removed
    for headline in headlines:
        if re.search('; .{1,20} \(.\)', headline) or re.search('\(.\)$', headline):
            logger.e('Headline may not be processed correctly:', headline[-100:])

    # Check for names
    regex_name = re.compile(r'([A-Z][a-z]+)\s+[A-Z]\.?\s+([A-Z][a-z]+)')
    regex_osurname = re.compile(r'O\'([A-Z][a-z]+)')
    for i, _ in enumerate(headlines):
        headlines[i] = regex_osurname.sub('\\1', headlines[i])
        headlines[i] = regex_name.sub('\\1 \\2', headlines[i])
        headlines[i] = headlines[i].replace('Steven', 'Steve')  # Steve and Steven Jobs

    # iPhone names
    regex_iphone = re.compile(r'i[Pp]hone\s+(\d\d|\dG?S?c?s?|SE|XR?S?)(\s+S)?(\s+(Plus|Max|Pro))?(\s+(Plus|Max|Pro))?')
    for i, headline in enumerate(headlines):
        headlines[i] = regex_iphone.sub('iphone', headline) # iPhone names

    # Remove some specific contraptions from some companies
    for i, _ in enumerate(headlines):
        if headlines[i].endswith('. S'):
            headlines[i] = headlines[i].replace('. S', '')
        headlines[i] = headlines[i].replace('À', 'A')
        headlines[i] = headlines[i].replace('A. A. C.', 'AAC')
        headlines[i] = headlines[i].replace('Toys \'R\' Us', 'ToysRUs')
        headlines[i] = headlines[i].replace('&#038;', '&')
        headlines[i] = headlines[i].replace('AT&T', 'ATT')
        headlines[i] = headlines[i].replace('S.&P.', 'SP')
        headlines[i] = headlines[i].replace('S&P.', 'SP')
        headlines[i] = headlines[i].replace('S&P', 'SP')
        headlines[i] = headlines[i].replace('R&B', 'RB')
        headlines[i] = headlines[i].replace('I&S', 'IS')
        headlines[i] = headlines[i].replace('J.P. Morgan', 'JP Morgan')
        headlines[i] = headlines[i].replace('J. P. Morgan', 'JP Morgan')
        headlines[i] = headlines[i].replace('J. P Morgan', 'JP Morgan')
        headlines[i] = headlines[i].replace('F.B.I.', 'FBI')
        headlines[i] = headlines[i].replace('hip-hop', 'hiphop')
        headlines[i] = headlines[i].replace('M&A', 'MA')
        headlines[i] = headlines[i].replace('M&M', 'MM')
        headlines[i] = headlines[i].replace('OS X', 'OSX')
        headlines[i] = headlines[i].replace('Verizon is V Cast service', 'Verizon')
        headlines[i] = headlines[i].replace('Galaxy S II', 'GalaxyS')
        headlines[i] = headlines[i].replace('Galaxy S III', 'GalaxyS')
        headlines[i] = headlines[i].replace('S II', 'GalaxyS')
        headlines[i] = headlines[i].replace('S III', 'GalaxyS')
        headlines[i] = headlines[i].replace('Nexus Q', 'NexusQ')
        headlines[i] = headlines[i].replace('J.& J.', 'JJ')
        headlines[i] = headlines[i].replace('Mac OX X', 'Mac')
        headlines[i] = headlines[i].replace('iTunes U', 'iTunes')
        headlines[i] = headlines[i].replace('Zen Vision:M', 'Zen Vision')
        headlines[i] = headlines[i].replace('Vision:M', 'Vision')
        headlines[i] = headlines[i].replace('Apple I', 'Apple Computer')
        headlines[i] = headlines[i].replace('Final Cut Pro X', 'VideoEditor')
        headlines[i] = headlines[i].replace('The Elder Scrolls V: Skyrim', 'Game')

        headlines[i] = headlines[i].replace('C.E.O.', 'CEO')
        headlines[i] = headlines[i].replace('iMac', 'Mac')
        headlines[i] = headlines[i].replace('Apple ComputerIE', 'Apple')
        headlines[i] = headlines[i].replace('Apple-1', 'Apple')
        headlines[i] = headlines[i].replace('Apple Computernc', 'Apple')
        headlines[i] = headlines[i].replace('Apple Computer', 'Apple')
        headlines[i] = headlines[i].replace('Apple Co', 'Apple')
        headlines[i] = headlines[i].replace('Apple Co.', 'Apple')
        headlines[i] = headlines[i].replace('Apple Company', 'Apple')
        headlines[i] = headlines[i].replace('Apple Corps Ltd.', 'Apple')
        headlines[i] = headlines[i].replace('Apple Corps', 'Apple')
        headlines[i] = headlines[i].replace('Apple Inc', 'Apple')
        headlines[i] = headlines[i].replace('Apple Inc.', 'Apple')
        headlines[i] = headlines[i].replace('Apple computers', 'Apple')
        headlines[i] = headlines[i].replace('Apple computer', 'Apple')
        headlines[i] = headlines[i].replace('Apple computers.', 'Apple')
        headlines[i] = headlines[i].replace('Apple computer.', 'Apple')

    # Remove single words in mayus
    for k, headline in enumerate(headlines):
        tokens, spans = tokens_span(headline)

        offset = 0
        for i, token in enumerate(tokens):
            if len(token) == 1 and token.isupper() and token.isalnum() and not is_subject_uppercase_i(tokens, i) and not is_preposition_uppercase_a(tokens, i):
                headlines[k] = headlines[k][:spans[i][0] + offset] + headlines[k][spans[i][1] + offset:]
                offset -= 1

        headlines[k] = regex_spaces.sub(' ', headlines[k])  # Remove multiple spaces

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
    return headlines


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
        json.dump(news, f)


if __name__ == '__main__':
    main()
