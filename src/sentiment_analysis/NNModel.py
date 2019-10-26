import keras
from keras.layers import Dense, LSTM


class NNModel(keras.Model):

    def __init__(self, embedding_layer):
        super().__init__(name='NewsModel')
        if embedding_layer:
            self.__embedding = embedding_layer
        self.__lstm = LSTM(100)
        self.__dense = Dense(1, activation='sigmoid')

    def call(self, inputs, mask=None):
        x = self.__embedding(inputs)
        x = self.__lstm(x)
        return self.__dense(x)
