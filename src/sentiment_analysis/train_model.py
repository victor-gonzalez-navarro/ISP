import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json
from datetime import datetime
from pathlib import Path
from gensim.models import KeyedVectors

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from utils import smooth, cache
from keras.preprocessing import sequence
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib

from sentiment_analysis.NewsModel import NewsModel, NewsModelEmbedding
from sentiment_analysis.BOWModel import BOWModel

QUERY = 'Apple'
FILE_PATH = Path(__file__).resolve().parents[0]
MODEL_PATH = (FILE_PATH / '../../models/news_model_google.h5')
GOOGLE_W2VEC_DB = (FILE_PATH / '../../data/GoogleNews-vectors-negative300.bin').resolve()
STOCK_PATH = (FILE_PATH / '../../data/sp500.csv').resolve()
NEWS_PATH = (FILE_PATH / '../../data/apple_20070101_20071231_1571064.json').resolve()
BEGIN_TIME = datetime(year=2007, month=1, day=1)
END_TIME = datetime(year=2008, month=1, day=4)
SMOOTH_SIZE = 3
MAX_INPUT_LEN = 150


def read_stocks(path: Path, start, end, normalize=True):
    """ Read stocks values from csv pick those beween start and end """
    data = pd.read_csv(path)
    data['Date'] = data['Date'].apply(lambda x: datetime.strptime(x, '%d-%m-%y'))
    x = data['Date'].tolist()
    y = data['Open'].tolist()
    idx_start = next(i for i, v in enumerate(x) if v >= start)
    idx_end = next(i for i, v in enumerate(x) if v >= end)
    x = x[idx_start:idx_end]
    y = y[idx_start:idx_end]
    if normalize:
        y = (np.array(y) - np.min(y)) / (np.max(y) - np.min(y))
    return x, y


def read_news(path: Path, query=None):
    """ """
    with path.open('r') as f:
        data = json.load(f)

    news = []
    for article in data:
        txt = article['abstract'] or article['snippet'] or article['lead_paragraph']
        if txt and (article['document_type'] == 'article') and (not query or query in txt):
            news.append({
                'type': 'article',
                'txt': txt,
                'date': datetime.strptime(article['pub_date'], '%Y-%m-%dT%H:%M:%S+%f'),
            })

    return news


def convert_news2vec(news, model, stop_words, max_size=73):
    for article in news:
        txt = article['txt']
        words = txt.replace("\n", " ")
        words = [word.lower() for word in word_tokenize(words) if word.isalpha()]
        words = [word for word in words if word not in stop_words]
        if isinstance(model, BOWModel):
            article['word2vec'] = [model[word] for word in words]
        else:
            article['word2vec'] = [model[word] for word in words if word in model.vocab]
        if len(article['word2vec']) > max_size:
            print('Article too long !:',  article['txt'])
            article['word2vec'] = article['word2vec'][:max_size]


def add_binary_label(x, y, news, time_window=0):
    for article in news:
        idx = next(i for i, v in enumerate(x) if v >= article['date'])
        start = max(0, idx - 10)
        end = min(len(y) - 1, idx + 10)
        diff = y[end] - y[start]
        article['outcome'] = 1 if diff >= 0 else 0


def init():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

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


def main():
    init()

    # --- Preprocess ---
    stop_words = set(stopwords.words('english'))
    word2vec_model = BOWModel()
    # word2vec_model = cache(lambda: KeyedVectors.load_word2vec_format(GOOGLE_W2VEC_DB, binary=True), path=Path('./word2vec_model.pickle'))

    x, y = read_stocks(STOCK_PATH, start=BEGIN_TIME, end=END_TIME)
    smooth_y = smooth(y, half_window=SMOOTH_SIZE)
    news = read_news(NEWS_PATH, query=QUERY)
    add_binary_label(x, y, news)
    convert_news2vec(news, word2vec_model, stop_words, MAX_INPUT_LEN)

    # --- Train ---
    x_train, y_train, x_test, y_test = gen_training(news, test_size=0.2, max_input_len=MAX_INPUT_LEN)

    model = NewsModelEmbedding(len(word2vec_model), 32, MAX_INPUT_LEN)
    # model = NewsModel()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)
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
