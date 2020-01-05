from pathlib import Path

PERIOD = 7  # Days before day d, where averages are taken
WINDOWS_SIZES = (2, 10, 20, 30)
WINDOW_PLOT_SIZE = WINDOWS_SIZES[1]
W2V_MODEL_NAME = 'glove-wiki-gigaword-50'

FILE_PATH = Path(__file__).resolve().parents[0]
SP500_PATH = (FILE_PATH / '../data/sp500.csv').resolve()
APPLE_PATH = (FILE_PATH / '../data/AAPL.csv').resolve()
UNRATE_PATH = (FILE_PATH / '../data/unrate.csv').resolve()
NEWS_PATH = (FILE_PATH / '../data/preprocessed_all.json').resolve()
NEWS_PREDICTED = (FILE_PATH / '../data/news_predicted.json').resolve()
PRODUCTS_PATH = (FILE_PATH / '../data/products.csv').resolve()
MODELS_PATH = (FILE_PATH / '../models/').resolve()