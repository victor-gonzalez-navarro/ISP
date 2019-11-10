import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from utils import color_gen
from sklearn import linear_model


class LSTM_Global:

    def __init__(self, config, name=None, model_path=None):
        if model_path is None:
            self.name = name
            self.model = Sequential(name=name)
        else:
            self.model = load_model(model_path)
            self.name = self.model.name
        self.config = config

    def buildLayers(self):
        #self.model.add(LSTM(self.config['layers_lstm_global']['neurons'][0],
        #                    input_shape=(self.config['layers_lstm_global']['timesteps'], 2),
        #                    return_sequences=True))
        #self.model.add(Dropout(self.config["layers_lstm_global"]["droprate"][0]))
        #self.model.add(LSTM(self.config['layers_lstm_global']['neurons'][1],
        #                    input_shape=(None, None), #(self.config['layers_lstm_global']['timesteps'], None),
        #                    return_sequences=True))
        #self.model.add(LSTM(self.config['layers_lstm_global']['neurons'][2],
        #                    input_shape=(None, None), #(self.config['layers_lstm_global']['timesteps'], None),
        #                    return_sequences=False))
        #self.model.add(Dropout(self.config["layers_lstm_global"]["droprate"][1]))
        #self.model.add(Dense(self.config['layers_lstm_global']['timesteps'],
        #                     activation=self.config['layers_lstm_global']['activationDense']))

        self.model.add(Dense(self.config['layers_lstm_global']['timesteps'],
                             input_shape=(1, self.config['layers_lstm_global']['timesteps'] * 2),
                             return_sequences=True))

        self.model.compile(loss=self.config['layers_lstm_global']['loss'],
                           optimizer=self.config['layers_lstm_global']['optimizer'])

    def fit(self, generator, n_seqs):
        callbacks = [
            ModelCheckpoint(filepath='../models/'+self.name, monitor='loss', save_best_only=True)
        ]

        steps_per_epoch = np.ceil(n_seqs/self.config['batch_size'])
        self.model.fit_generator(generator=generator, steps_per_epoch=steps_per_epoch,
                                 epochs=self.config['num_epochs'], callbacks=callbacks, workers=1)

    def fit_reg(self, trn_seqs):
        pass

    def predict(self, tst_data, plot=None):
        y_pred = np.zeros((len(tst_data), len(tst_data[0][1])))
        y_pred_reg = np.copy(y_pred)

        color = color_gen()
        for seq_id in range(len(tst_data)):
            y_pred[seq_id, :] = self.model.predict(np.array([tst_data[seq_id][0]]))

            # Create linear regression object
            regr = linear_model.LinearRegression()
            regr.fit(np.array(range(y_pred.shape[1])).reshape(y_pred.shape[1], 1), y_pred[seq_id, :])
            y_pred_reg[seq_id, :] = regr.predict(np.array(range(y_pred.shape[1])).reshape(y_pred.shape[1], 1))

            if plot is not None:
                start = seq_id * y_pred.shape[1]
                end = start + y_pred.shape[1]
                plot.plot(list(range(start, end)), y_pred[seq_id, :], next(color) + '^-', markersize=3,
                          linewidth=0.5)
                plot.plot(list(range(start, end)), y_pred_reg[seq_id, :], next(color) + '^-', markersize=3,
                          linewidth=0.5)

        return y_pred_reg
