import keras
from keras.layers import Embedding, Dense, LSTM


class NewsModelEmbedding(keras.Model):

    def __init__(self, vocab_len, embedding_vec_len, in_len):
        super().__init__(name='NewsModelEmbedding')
        self.__embedding = Embedding(vocab_len, embedding_vec_len, input_length=in_len)
        self.__lstm = LSTM(100)
        self.__dense = Dense(1, activation='sigmoid')

    def call(self, inputs, mask=None):
        x = self.__embedding(inputs)
        x = self.__lstm(x)
        return self.__dense(x)


class NewsModel(keras.Model):

    def __init__(self):
        super().__init__(name='NewsModel')
        #self.__dense1 = Dense(128, activation='sigmoid')
        #self.__dense2 = Dense(32, activation='sigmoid')
        self.__lstm = LSTM(128)
        self.__dense3 = Dense(1, activation='sigmoid')

    def call(self, inputs, mask=None):
        #x = self.__dense1(inputs)
        #x = self.__dense2(x)
        x = self.__lstm(inputs)
        return self.__dense3(x)
