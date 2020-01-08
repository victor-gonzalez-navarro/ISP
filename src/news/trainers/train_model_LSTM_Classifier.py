import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import json

from src.news.models.w2v_models import GensimModel
from keras.preprocessing import sequence
from tensorflow.python.client import device_lib
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from src.news.news_utils import read_news_all, prune_news, add_labels
from src.news.stock_utils import load_all
from src.news.common import SP500_PATH, APPLE_PATH, PRODUCTS_PATH, NEWS_PATH, MODELS_PATH, W2V_MODEL_NAME, NEWS_PREDICTED, WINDOWS_SIZES, TRAIN
from src.news.trainers.train_model_LSTM_Regressor import rebalance, gen_training
from src.news.models.NNModel import NNModel

from pandas.plotting import register_matplotlib_converters


ATTRIBUTE = 'word_indexes'
DDBB = ['nyt']
MAX_INPUT_LEN = 80
SMOOTH_SIZE = 10
PLOT = True
SAVE = False


def train_lstm_classifier(news, test_size, window_size, word2vec_model):
    """ Train a LSTM Regressor model """
    x_train, \
    y_train, \
    x_test, \
    y_test = gen_training(news, test_size, MAX_INPUT_LEN, window_size, ATTRIBUTE, has_embeddings=True)

    model = NNModel(word2vec_model.generate_embedding_layer(MAX_INPUT_LEN, trainable=False))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=8, batch_size=64)
    return model, history


def labels_to_binary(news):
    for i, article in enumerate(news):
        for window_size in WINDOWS_SIZES:
            news[i]['windows'][window_size] = 1.0 if news[i]['windows'][window_size] >= 1.0 else 0.0
    return news


def main():
    # LSTM / Classifier
    # --- Load ---
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
    news = labels_to_binary(news)

    # --- Train ---
    train_date = apple_x[int(len(apple_x) * TRAIN)]
    test_size = next(i for i, article in enumerate(news) if article['date'] >= train_date)  # news[:test_size] is training

    models = []
    for window_size in WINDOWS_SIZES:
        model, history = train_lstm_classifier(news, test_size, window_size, word2vec_model)
        model.save_weights(str(MODELS_PATH / 'news_model_lstm_classifier_ws{:d}.h5'.format(window_size)))
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
            y_true = []
            y_pred = []
            for k, article in enumerate(tqdm(news)):
                y_true.append('pos' if article['real_labels'][i] == 1.0 else 'neg')
                y_pred.append('pos' if article['predicted_labels'][i] == 1.0 else 'neg')
            conf = confusion_matrix(y_true, y_pred, labels=['pos', 'neg'])
            print(conf)

    if SAVE:
        with NEWS_PREDICTED.open('w') as f:
            json.dump(modified_news, f, indent=4, default=str)


if __name__ == '__main__':
    register_matplotlib_converters()
    print('Using ', device_lib.list_local_devices()[-1].name)
    main()
