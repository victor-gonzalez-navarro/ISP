import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

from src.news.common import NEWS_PREDICTED, WINDOWS_SIZES


def main():
    with NEWS_PREDICTED.open('r') as f:
        news = json.load(f)

    for i, _ in enumerate(WINDOWS_SIZES):
        x = []
        y = []
        for k, article in enumerate(news):
            x.append(article['real_labels'][i])
            y.append(article['predicted_labels'][i])

        x = np.array(x)
        y = np.array(y)
        model = LinearRegression().fit(x.reshape(-1, 1), y)
        r_sq = model.score(x.reshape(-1, 1), y)
        y_line = model.predict(x.reshape(-1, 1))

        plt.subplot(2, 2, i + 1)
        plt.scatter(x, y, alpha=0.5)
        plt.plot(x, y_line, color='orange', linewidth=3, alpha=0.8)
        plt.xlabel('Real')
        plt.ylabel('Predicted')
        plt.gca().set_title('Pearson Corr Coef {:.3f}'.format(np.corrcoef(x, y)[0, 1]))
    plt.show()


if __name__ == '__main__':
    main()
