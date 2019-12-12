import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint


class MLP_Network:
    def __init__(self, config, name=None, model_path=None):
        if model_path is None:
            self.name = name
            self.model = Sequential(name=name)
        else:
            self.model = load_model(model_path)
            self.name = self.model.name
        self.config = config['mlp']

    def buildLayers(self, input_feats):
        first = True
        for layer in self.config['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['droprate'] if 'droprate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            # return_seq = layer['return_seq'] if 'return_seq' in layer else None
            if first:
                input_dim = (input_feats,)
                first = False
            else:
                input_dim = (None, None)

            if layer['type'] == 'Dense':
                self.model.add(Dense(neurons,
                                     input_shape=input_dim,
                                     activation=activation))
            elif layer['type'] == 'Dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=self.config['loss'], optimizer=self.config['optimizer'])

    def fit(self, X, Y):
        callbacks = [
            ModelCheckpoint(filepath='../models/' + self.name, monitor='loss', save_best_only=True)
        ]

        # steps_per_epoch = int(np.ceil(len(Y) / self.config['batch_size']))
        self.model.fit(x=np.array(X), y=Y,  # steps_per_epoch=steps_per_epoch,
                       epochs=self.config['num_epochs'], callbacks=callbacks)

    def fit2(self, gen, n_seqs):
        callbacks = [
            ModelCheckpoint(filepath='../models/' + self.name, monitor='loss', save_best_only=True)
        ]
        steps_per_epoch = np.ceil(n_seqs / self.config['batch_size'])
        self.model.fit_generator(generator=gen, steps_per_epoch=steps_per_epoch,
                                 epochs=self.config['num_epochs'], callbacks=callbacks, workers=1)

    def predict(self, X_tst):
        return self.model.predict(X_tst)
