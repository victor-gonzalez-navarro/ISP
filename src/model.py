import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint

class Model:

    def __init__(self, name, config, model=None):
        if model is None:
            self.name = name
            self.model = Sequential()
        else:
            self.model = model
        self.config = config

    def buildLayers(self):
        a = (self.config['layers']['timesteps'], len(self.config['features']))
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

        self.model.fit_generator(generator=generator, steps_per_epoch=int(n_seqs/self.config['batch_size']),
                                 epochs=self.config['num_epochs'], callbacks=callbacks, workers=1)

    def predict(self):
        self.model.predict()
