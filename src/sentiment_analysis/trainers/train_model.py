import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import json

from sentiment_analysis.w2v_models import GensimModel
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import sequence
from tensorflow.python.client import device_lib
from tqdm import tqdm
from src.news_utils import read_news_all, prune_news, add_labels
from src.stock_utils import load_all
from src.common import SP500_PATH, APPLE_PATH, PRODUCTS_PATH, NEWS_PATH, MODELS_PATH, W2V_MODEL_NAME, NEWS_PREDICTED, WINDOWS_SIZES

from sentiment_analysis.NNModel import NNModel

from pandas.plotting import register_matplotlib_converters


ATTRIBUTE = 'word_indexes'
DDBB = ['nyt']
TRAIN = 0.85
MAX_INPUT_LEN = 80
SMOOTH_SIZE = 10
PLOT = False


def init():
    register_matplotlib_converters()
    print('Using ', device_lib.list_local_devices()[-1].name)


def rebalance(news, rebalance_limit=0.05):
    """ If there are too many news of one class, remove some so they became 50/50 """
    pos_samples = []
    neg_samples = []

    for article in news:
        avg_val = np.average([v for _, v in article['windows'].items()])

        if avg_val >= 1.0:
            pos_samples.append(article)
        else:
            neg_samples.append(article)

    pos_percentage = len(pos_samples) / len(news)
    neg_percentage = len(neg_samples) / len(news)
    print('Positive samples: {:.1f}% ({:d} samples)'.format(pos_percentage * 100.0, len(pos_samples)))
    print('Negative samples: {:.1f}% ({:d} samples)'.format(neg_percentage * 100.0, len(neg_samples)))
    if rebalance and abs(pos_percentage - neg_percentage) >= rebalance_limit:
        pop_size = min(len(pos_samples), len(neg_samples))
        rebalanced_news = random.sample(pos_samples, k=pop_size) + random.sample(neg_samples, k=pop_size)
        print('Rebalanced to 50/50 ({:d} samples each)'.format(pop_size))
        return sorted(rebalanced_news, key=lambda x: x['date'])

    return news


def gen_training(news, test_size, max_input_len, window_size, attribute, has_embeddings=True):
    assert attribute in ['word_indexes', 'word_vector', 'word_indexes_title', 'word_vector_title']

    train = [(article[attribute], article['windows'][window_size]) for article in news[:test_size]]
    test = [(article[attribute], article['windows'][window_size]) for article in news[test_size:]]
    # (X, Y)
    # X => list of words or indexes
    # Y => Percentage change for that window

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


def train(news, test_size, window_size, word2vec_model):
    x_train, \
    y_train, \
    x_test, \
    y_test = gen_training(news, test_size, MAX_INPUT_LEN, window_size, ATTRIBUTE, has_embeddings=True)

    model = NNModel(word2vec_model.generate_embedding_layer(MAX_INPUT_LEN, trainable=False))
    model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=8, batch_size=64)
    return model, history


def main():
    # LSTM / Regressor
    # --- Load ---
    init()
    sp500_x, apple_x, \
    sp500_y, apple_y, \
    percent_sp500_y, percent_apple_y, \
    correct_apple_y, diff, items = load_all(SMOOTH_SIZE, PRODUCTS_PATH, SP500_PATH, APPLE_PATH)
    news = read_news_all(NEWS_PATH, date_attribute='date', date_format='%Y-%m-%d %H:%M:%S', sort_date=True, databases=DDBB)
    word2vec_model = GensimModel(W2V_MODEL_NAME)

    # --- Prepare ---
    news = prune_news(news, max_date=apple_x[-1])
    news = add_labels(apple_x, apple_y, news)
    news = rebalance(news)

    # --- Train ---
    train_date = apple_x[int(len(apple_x) * TRAIN)]
    test_size = next(i for i, article in enumerate(news) if article['date'] >= train_date)  # news[:test_size] is training

    models = []
    for window_size in WINDOWS_SIZES:
        model, history = train(news, test_size, window_size, word2vec_model)
        model.save_weights(str(MODELS_PATH / 'news_model_ws{:d}.h5'.format(window_size)))
        models.append(model)

    # --- Test --- (Predict)
    predictions = []
    X = [article['word_indexes'] for article in news]
    for model in tqdm(models, desc='Predicting with each model window'):
        predictions.append(model.predict(sequence.pad_sequences(X, dtype=object, maxlen=MAX_INPUT_LEN)))

    modified_news = []
    for k, article in enumerate(tqdm(news)):
        article['real_labels'] = [v for _, v in article['windows'].items()]
        article['predicted_labels'] = [float(predictions[i][k][0]) for i in range(len(WINDOWS_SIZES))]
        modified_news.append(article)

    # --- Test --- (Display)
    if PLOT:
        for i, _ in enumerate(WINDOWS_SIZES):
            for k, article in enumerate(tqdm(news)):
                plt.scatter(article['real_labels'][i], article['predicted_labels'][i])
            plt.show()

    with NEWS_PREDICTED.open('w') as f:
        json.dump(modified_news, f, indent=4, default=str)


if __name__ == '__main__':
    main()
