import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import re
import json
import nltk
from pathlib import Path
from tqdm import tqdm
from sentiment_analysis.w2v_models import GensimModel
from tensorflow.python.client import device_lib

from src.sentiment_analysis.ground_truth import read_news

FILE_PATH = Path(__file__).resolve().parents[0]
IN_PATH = (FILE_PATH / '../../data/preprocessed_wsj6.json').resolve()
OUT_PATH = (FILE_PATH / '../../data/preprocessed_wsj6_.json').resolve()
STOP_WORDS_PATH = (FILE_PATH / '../../data/stop_words.txt').resolve()
MAX_LEN = 5000


def init():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    print('Using ', device_lib.list_local_devices()[-1].name)


def print_lines(txt, line_length=90):
    while len(txt) > line_length:
        if txt[0] == ' ':
            txt = txt.strip()
        else:
            length = line_length + txt[line_length:].find(' ')
            print(txt[:length])
            txt = txt[length:]
    print(txt.strip())


def clean_paragraphs(paragraphs):
    if len(paragraphs) == 1:
        return paragraphs[0]

    else:
        content_paragraphs = []
        author = None

        for p in paragraphs:
            if p == 'Advertisement':
                continue

            result = re.search(r'^\s+By\s+<span.*?data-byline-name="(.*?)"', p)
            if result:
                author = result.group(1)
                continue

            if author and p == author:
                break

            content_paragraphs.append(p)

        content = ' '.join(content_paragraphs).strip()
        content = re.sub(r'\s+', ' ', content)
        return content

def get_content(article):
    if article['p'] is not None:
        return clean_paragraphs(article['p'])
    elif len(article['lead_paragraph']) < 20:
        return article['txt']
    else:
        return article['lead_paragraph']


def accept_token(token):
    return token.isalpha()


def vectorize(content, stop_words):
    if len(content) > MAX_LEN:
        content = content[:MAX_LEN]
    tokens = [token.lower() for token in nltk.word_tokenize(content)]
    result = [token for token in tokens if token not in stop_words and accept_token(token)]
    return result


def indexes(tokens, word2vec_model):
    indexes = []
    for token in tokens:

        if not token.isalnum():
            continue

        try:
            vec = word2vec_model.word2index(token)
            indexes.append(vec)
        except KeyError:
            pass

    return indexes


def load_words(path: Path):
    words = []
    with path.open('r') as f:
        for line in f:
            words.append(line.strip().lower())

    return words


def main():
    init()

    stop_words = set(nltk.corpus.stopwords.words('english'))# + load_words(STOP_WORDS_PATH))

    modified_stop_words = []
    for s in stop_words:
        if '\'' in s:
            modified_stop_words.extend(nltk.word_tokenize(s))
        else:
            modified_stop_words.append(s)

    stop_words = set([word.lower() for word in modified_stop_words])

    # Negative and positive stopwords
    stop_words.remove('will')
    stop_words.remove('against')

    # W2V
    print('Loading model...')
    word2vec_model = GensimModel('glove-wiki-gigaword-50')  # BOWModel() # 2513 wiki
    print('Done')

    news = read_news(IN_PATH)

    for i, article in enumerate(tqdm(news)):
        content = article['content']
        article['word_vector'] = vectorize(content, stop_words)
        article['word_indexes_big'] = indexes(article['word_vector'], word2vec_model)

    with OUT_PATH.open('w') as f:
        json.dump(news, f, indent=4, sort_keys=True, default=str)


if __name__ == '__main__':
    main()
