import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from datetime import datetime
from pathlib import Path
import random

from sentiment_analysis.w2v_models import GensimModel
import matplotlib.pyplot as plt
import numpy as np
from utils import smooth
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from src.sentiment_analysis.ground_truth import read_news, read_stocks, prune_news, add_labels

from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import _tree
from sentiment_analysis.NNModel import NNModel

from pandas.plotting import register_matplotlib_converters


FILE_PATH = Path(__file__).resolve().parents[0]
IN_PATH = (FILE_PATH / '../../data/preprocessed.json').resolve()
STOCK_PATH = (FILE_PATH / '../../data/sp500.csv').resolve()
MODEL_PATH = (FILE_PATH / '../../models/news_model.h5').resolve()

BEGIN_TIME = datetime(year=2007, month=1, day=1)
END_TIME = datetime(year=2008, month=1, day=4)

MAX_INPUT_LEN = 80
W2V_MODEL_NAME = 'glove-wiki-gigaword-100'
SMOOTH_SIZE = 3
PLOT = False
PERIOD = 7  # Days before day d, where averages are taken
WINDOWS_SIZES = (2, 10, 20, 30)
WINDOW_PLOT_SIZE = WINDOWS_SIZES[1]


def init():
    register_matplotlib_converters()
    print('Using ', device_lib.list_local_devices()[-1].name)


def rebalance(news, rebalance_limit=0.05, multiplier=1):
    pos_samples = []
    neg_samples = []

    for article in news:
        avg_val = np.average([article[next_window] for next_window in article['windows']])

        if avg_val >= 1.0 * multiplier:
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


def unison_shuffled_copies(a, b):
    assert a.shape[0] == b.shape[0]
    p = np.random.permutation(a.shape[0])
    return a[p], b[p]


def gen_training_tree(X, Y, test_size):
    assert X.shape[0] == Y.shape[0]

    X, Y = unison_shuffled_copies(X, Y)
    length = X.shape[0]
    train_size = length - int(test_size * length)

    X_train = X[:train_size]
    X_test = X[train_size:]
    Y_train = Y[:train_size]
    Y_test = Y[train_size:]
    return X_train, Y_train, X_test, Y_test


def gen_training(news, test_size, max_input_len, has_embeddings=True):
    data = [(article['word_indexes'], np.clip(article[WINDOWS_SIZES[0]], 0.95, 1.05) * 10 - 10) for article in news]
    # data = [(article['word_indexes'], 0 if article[WINDOWS_SIZES[0]] < 1.0 else 1.0) for article in news]
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


def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    print("def tree({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, threshold))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, threshold))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)


def main():
    init()

    x, y = read_stocks(STOCK_PATH)
    news = read_news(IN_PATH)
    smooth_y = smooth(y, half_window=SMOOTH_SIZE)
    news = prune_news(news, max_date=x[-1])
    news = add_labels(x, smooth_y, news, multiplier=100, limits=10)
    # news = rebalance(news, multiplier=100)
    #  plot_ground_truth_per_article(x, smooth_y, news)

    import nltk
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    for article in news:
        if article[news[0]['windows'][1]] < 95:
            print(article['txt'])
    exit()

    for window_size in news[0]['windows']:
        Y_train = []
        predicted_y = []
        for article in tqdm(news):
            sid = SentimentIntensityAnalyzer()
            predicted_y.append(sid.polarity_scores(article['txt'])['compound'])
            Y_train.append(article[window_size])

        diff = [(predicted_y[i] - Y_train[i]) for i in range(len(Y_train))]
        plt.figure()
        plt.title('WS: ' + str(window_size))
        plt.xlabel('Predicted')
        plt.ylabel('Real')
        plt.scatter(predicted_y, Y_train, c=diff, alpha=0.8)
        plt.show()

    exit()

    # ---------------------------------------------------------------------------
    corpus = [' '.join(article['word_list']) for article in news]
    vectorizer = CountVectorizer(max_features=1000)
    window_size = news[0]['windows'][0]
    print('Window size: ', window_size)

    X = vectorizer.fit_transform(corpus)
    Y = np.array([article[window_size] for article in news])

    print(vectorizer.get_feature_names())
    print(X.shape)
    print(Y.shape)

    X_train, Y_train, X_test, Y_test = gen_training_tree(X, Y, test_size=0.2)

    # clf = tree.DecisionTreeRegressor()
    # clf = tree.DecisionTreeClassifier()
    clf = RandomForestClassifier(max_depth=128, random_state=0)
    clf = clf.fit(X_train, Y_train)

    print(Y_train.shape, Y_test.shape)

    predicted_y = clf.predict(X_train)
    #diff = [(predicted_y[i] - Y_train[i]) for i in range(len(Y_train))]
    #plt.figure()
    #plt.title('Train')
    #plt.xlabel('Predicted')
    #plt.ylabel('Real')
    #plt.scatter(predicted_y, Y_train, c=diff, alpha=0.8)
    #plt.show()

    print(confusion_matrix(predicted_y, Y_train))

    predicted_y = clf.predict(X_test)
    #diff = [(predicted_y[i] - Y_test[i]) for i in range(len(Y_test))]
    #plt.figure()
    #plt.title('Test')
    #plt.xlabel('Predicted')
    #plt.ylabel('Real')
    #plt.scatter(predicted_y, Y_test, c=diff, alpha=0.8)
    #plt.show()

    print(confusion_matrix(predicted_y, Y_test))

    # tree_to_code(clf, ['a', 'b'])
    exit()
    # --- Train ---
    word2vec_model = GensimModel(W2V_MODEL_NAME)  # BOWModel()
    x_train, y_train, x_test, y_test = gen_training(news, test_size=0.3, max_input_len=MAX_INPUT_LEN)

    model = NNModel(word2vec_model.generate_embedding_layer(MAX_INPUT_LEN, trainable=False))
    model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])
    #  model.load_weights(MODEL_PATH)
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=150, batch_size=64)
    #  model.save_weights(MODEL_PATH)

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss (mse)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    predicted_y = model.predict(x_train)
    diff = [(predicted_y[i] - y_train[i]) for i in range(len(y_train))]
    plt.figure()
    plt.scatter(predicted_y, y_train, c=diff, alpha=0.8)
    axes = plt.gca()
    axes.set_xlim([-0.2, 0.2])
    axes.set_ylim([-0.2, 0.2])
    plt.show()


if __name__ == '__main__':
    main()
