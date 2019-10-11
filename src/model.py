import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from utils import color_gen


class Model:

    def __init__(self, config, name=None, model_path=None):
        if model_path is None:
            self.name = name
            self.model = Sequential(name=name)
        else:
            self.model = load_model(model_path)
            self.name = self.model.name
        self.config = config

    def buildLayers(self):
        self.model.add(LSTM(self.config['layers']['neurons'][0],
                            input_shape=(self.config['layers']['timesteps'], len(self.config['features'])),
                            return_sequences=True))
        self.model.add(Dropout(self.config["layers"]["droprate"][0]))
        self.model.add(LSTM(self.config['layers']['neurons'][1],
                            input_shape=(self.config['layers']['timesteps'], len(self.config['features'])),
                            return_sequences=True))
        self.model.add(LSTM(self.config['layers']['neurons'][2],
                            input_shape=(self.config['layers']['timesteps'], len(self.config['features'])),
                            return_sequences=False))
        self.model.add(Dropout(self.config["layers"]["droprate"][1]))
        self.model.add(Dense(self.config['layers']['neurons'][3], activation=self.config['layers']['activationDense']))

        self.model.compile(loss=self.config['layers']['loss'], optimizer=self.config['layers']['optimizer'])

    def fit(self, generator, n_seqs):
        callbacks = [
            ModelCheckpoint(filepath='../models/'+self.name, monitor='loss', save_best_only=True)
        ]

        steps_per_epoch = np.ceil(n_seqs/self.config['batch_size'])
        self.model.fit_generator(generator=generator, steps_per_epoch=steps_per_epoch,
                                 epochs=self.config['num_epochs'], callbacks=callbacks, workers=1)

    def predict(self, tst_data, next_k_items=50, plot=None):

        y_pred = np.zeros((next_k_items, 1))
        counter = 0
        color = color_gen()

        for seq_id in range(len(tst_data)):
            if counter % next_k_items == 0:
                counter = 0
                sequence = np.copy(tst_data[seq_id][0])

            else:
                sequence = np.concatenate((tst_data[seq_id][0][:-counter, :],
                                           np.concatenate((y_pred[:counter], y_pred[:counter]), axis=1)),
                                          axis=0)

            y_pred[counter] = self.model.predict(np.array([sequence]))[0, 0]

            counter += 1

            if counter == next_k_items and plot is not None:
                end = seq_id + self.config['length_sequence'] - next_k_items
                start = end - next_k_items
                plot.plot(list(range(start, end)), y_pred[:, 0], next(color), linewidth=0.5)

