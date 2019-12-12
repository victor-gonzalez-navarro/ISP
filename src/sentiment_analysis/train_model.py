import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import random

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import langid
from utils import smooth
from keras.preprocessing import sequence
from nltk.classify import textcat
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib

from sentiment_analysis.NNModel import NNModel
from sentiment_analysis.w2v_models import GensimModel

QUERY = 'Apple'
FILE_PATH = Path(__file__).resolve().parents[0]
REDDIT_PATH = (FILE_PATH / '../../data/reddit').resolve()
NYT_PATH = (FILE_PATH / '../../data/nyt').resolve()
MODEL_PATH = (FILE_PATH / '../../models/news_model.h5')
GOOGLE_W2VEC_DB = (FILE_PATH / '../../data/GoogleNews-vectors-negative300.bin').resolve()
STOCK_PATH = (FILE_PATH / '../../data/sp500.csv').resolve()
BEGIN_TIME = datetime(year=2007, month=1, day=1)
END_TIME = datetime(year=2008, month=1, day=4)
SMOOTH_SIZE = 3
MAX_INPUT_LEN = 150
PREV_WINDOW = 1
NEXT_WINDOW = 7


def read_stocks(path: Path, start=None, end=None, normalize=True):
    """ Read stocks values from csv pick those between start and end """
    data = pd.read_csv(path)
    data['Date'] = data['Date'].apply(lambda x: datetime.strptime(x, '%d-%m-%y'))
    x = data['Date'].tolist()
    y = data['Open'].tolist()
    if start is not None and end is not None:
        idx_start = next(i for i, v in enumerate(x) if v >= start)
        idx_end = next(i for i, v in enumerate(x) if v >= end)
        x = x[idx_start:idx_end]
        y = y[idx_start:idx_end]
    if normalize:
        y = (np.array(y) - np.min(y)) / (np.max(y) - np.min(y))
    return x, y


def read_news(datasets=None, sort=True):
    if datasets is None:
        datasets = ['reddit', 'nyt']

    news = []
    if 'reddit' in datasets:
        for file in REDDIT_PATH.iterdir():
            if file.is_file():
                with file.open('r') as f:
                    data = json.load(f)
                    data = data["data"]
                for article in data:
                    news.append({
                        'txt': article['title'].lower(),
                        'date': datetime.utcfromtimestamp(article['created_utc']),
                    })

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
                            'txt': txt.lower(),
                            'date': datetime.strptime(article['pub_date'], '%Y-%m-%dT%H:%M:%S+%f'),
                        })

    print(len(news))

    if sort:
        news = sorted(news, key=lambda x: x['date'])

    return news


def convert_news2vec(news, model, stop_words, max_size=73):
    for article in tqdm(news, desc='word2vec'):
        txt = article['txt']
        words = txt.replace("\n", " ")
        words = [word for word in word_tokenize(words) if word.isalpha()]
        #words = [word for word in words if word not in stop_words]
        article['word2vec'] = model.words2index(words)
        if len(article['word2vec']) > max_size:
            print('Article too long !:',  article['txt'])
            article['word2vec'] = article['word2vec'][:max_size]


def add_binary_label(x, y, news, absolute=True, rebalance=True, rebalance_limit=0.05):
    assert all(news[i]['date'] <= news[i + 1]['date'] for i in range(len(news) - 1))  # Assert sorted

    pos_examples = []
    neg_examples = []

    offset = 0
    for article in tqdm(news, desc='Adding ground truth labels'):
        idx = next(i for i, v in enumerate(x[offset:]) if v >= article['date']) + offset
        start = max(0, idx - PREV_WINDOW)
        end = min(len(y) - 1, idx + NEXT_WINDOW)
        diff = y[end] - y[start]
        if absolute:
            article['outcome'] = 1 if diff >= 0 else 0
        else:
            article['outcome'] = diff
        if diff > 0:
            pos_examples.append(article)
        elif diff < 0:
            neg_examples.append(article)
        offset = idx

    if not absolute:
        outcomes = [article['outcome'] for article in news]
        limit = max(abs(max(outcomes)), abs(min(outcomes)))
        for article in news:
            article['outcome'] = (limit + article['outcome'])  / limit - 1

    pos_percentage = len(pos_examples) / len(news)
    neg_percentage = len(neg_examples) / len(news)
    print('Positive samples: {:.1f}% ({:d} samples)'.format(pos_percentage * 100.0, len(pos_examples)))
    print('Negative samples: {:.1f}% ({:d} samples)'.format(neg_percentage * 100.0, len(neg_examples)))
    if rebalance and abs(pos_percentage - neg_percentage) >= rebalance_limit:
        pop_size = min(len(pos_examples), len(neg_examples))
        rebalanced_news = random.sample(pos_examples, k=pop_size) + random.sample(neg_examples, k=pop_size)
        print('Rebalanced to 50/50 ({:d} samples each)'.format(pop_size))
        return sorted(rebalanced_news, key=lambda x: x['date'])

    return news


def init():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/crubadan')
    except LookupError:
        nltk.download('crubadan')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    print('Using ', device_lib.list_local_devices()[-1].name)


def gen_training(news, test_size, max_input_len, has_embeddings=True):
    data = [(article['word2vec'], article['outcome']) for article in news]
    train, test = train_test_split(data, test_size=test_size)
    x_train, y_train = list(zip(*train))
    x_test, y_test = list(zip(*test))
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    if has_embeddings:
        x_train = sequence.pad_sequences(x_train, dtype=object, maxlen=max_input_len)
        x_test = sequence.pad_sequences(x_test, dtype=object, maxlen=max_input_len)
    else:
        x_train = sequence.pad_sequences(x_train, maxlen=max_input_len)
        x_test = sequence.pad_sequences(x_test, maxlen=max_input_len)
    return x_train, y_train, x_test, y_test


def findy(x, y, d):
    idx = 0
    while x[idx] <= d:
        idx += 1
    return y[idx]


def plot_ground_truth(x, y, news):
    select_color = lambda x: 'r' if x < 0 else 'g'
    labels = [article['outcome'] for article in news]
    labels1 = [findy(x, y, article['date']) for article in news]
    dates  = [article['date'] for article in news]
    colors = [select_color(i) for i in labels]

    plt.figure()
    plt.plot(x, y, alpha=0.8, label='%s Stock' % QUERY, zorder=0)
    plt.scatter(dates, labels1, alpha=0.8, s=4, c=colors, zorder=1)
    plt.xticks(rotation=30)
    plt.legend()
    plt.show()


def language_ratios(tokens):
    ratios = {}

    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(tokens)
        common_elements = words_set.intersection(stopwords_set)
        ratios[language] = len(common_elements)

    print(ratios)


def classify(identifier, text):
    fv = identifier.instance2fv(text)
    probs = identifier.norm_probs(identifier.nb_classprobs(fv))
    return {label: probs[i] for i, label in enumerate(identifier.nb_classes)}


def preprocess(news, stop_words):
    # Language
    # Spell corrector
    # Lexicon normalization
    # Stemming

    processed = []

    identifier = langid.langid.LanguageIdentifier.from_modelstring(langid.langid.model, norm_probs=True)

    for article in tqdm(news):
        txt = article['txt']
        lan = classify(identifier, txt)

        if lan['en'] < 0.0001:
            continue

        txt = txt.lower().replace("\n", " ")
        tokens = [word for word in word_tokenize(txt) if word.isalpha() and word not in stop_words]
        article['tokens'] = tokens
        processed.append(article)

    return processed


def main():
    init()

    # --- Preprocess ---
    stop_words = set(stopwords.words('english'))
    word2vec_model = GensimModel('glove-wiki-gigaword-100')  # BOWModel() # 2513 wiki

    x, y = read_stocks(STOCK_PATH)
    smooth_y = smooth(y, half_window=SMOOTH_SIZE)
    news = read_news(datasets=['nyt'])
    news = preprocess(news, stop_words)
    news = add_binary_label(x, smooth_y, news, absolute=False)
    convert_news2vec(news, word2vec_model, stop_words, MAX_INPUT_LEN)
    plot_ground_truth(x, smooth_y, news)
    exit()

    # --- Train ---
    x_train, y_train, x_test, y_test = gen_training(news, test_size=0.3, max_input_len=MAX_INPUT_LEN)

    model = NNModel(word2vec_model.generate_embedding_layer(MAX_INPUT_LEN, trainable=False))
    model.compile(loss='poisson', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=64)
    model.save_weights(MODEL_PATH)

    # --- Print results ---
    to_predict = [article['word2vec'] for article in news]
    to_predict = sequence.pad_sequences(to_predict, maxlen=MAX_INPUT_LEN)
    predicted = model.predict(to_predict)
    predicted = [i[0] for i in predicted]
    dates = [article['date'] for article in news]
    select_color = lambda x: 'r' if x < 0.5 else 'g'
    color = [select_color(i) for i in predicted]

    plt.figure()
    plt.plot(x, y, label='%s Stock' % QUERY)
    plt.plot(x, smooth_y, label='%s Stock smooth' % QUERY, alpha=0.8)
    plt.scatter(dates, predicted, alpha=0.5, s=2, c=color)
    predicted = [abs((i - 0.5) * 0.2) for i in predicted]
    plt.bar(dates, predicted, color=color)
    plt.xticks(rotation=30)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
