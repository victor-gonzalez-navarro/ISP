import keras
from keras.layers import Dense, LSTM, Dropout


class NNModel(keras.Model):

    def __init__(self, embedding_layer):
        super().__init__(name='NewsModel')
        if embedding_layer:
            self.__embedding = embedding_layer
        self.__lstm = LSTM(100)
        self.__dense0 = Dense(100, activation='relu')
        self.__drop = Dropout(0.2)
        self.__dense2 = Dense(1)#, activation='relu')

    def call(self, inputs, mask=None):
        x = self.__embedding(inputs)
        x = self.__lstm(x)
        x = self.__dense0(x)
        x = self.__drop(x)
        return self.__dense2(x)
