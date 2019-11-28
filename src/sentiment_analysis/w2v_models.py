from abc import ABC, abstractmethod

import gensim.downloader as api
from keras.layers import Embedding


class W2VModel(ABC):

    @abstractmethod
    def word2index(self, word):
        pass

    @abstractmethod
    def word2vec(self, word_list):
        pass

    def words2index(self, word_list):
        # TODO company and people names should be preprocessed
        indexes = []
        for word in word_list:
            try:
                indexes.append(self.word2index(word))
            except KeyError:
                print('Word \'%s\' not in voc' % word)
        return indexes

    def words2vec(self, word_list):
        # TODO company and people names should be preprocessed
        # result = []
        # error = False
        # for word in word_list:
        #     try:
        #        result.append(self.word2vec(word))
        #    except KeyError:
        #        print('Word \'%s\' not in voc' % word)
        #        error = True
        # return result, error
        vec = []
        for word in word_list:
            try:
                vec.append(self.word2vec(word))
            except KeyError:
                print('Word \'%s\' not in voc' % word)
        return vec

    @abstractmethod
    def generate_embedding_layer(self, in_len):
        pass

    def __getitem__(self, key):
        return self.word2vec(key)


class GensimModel(W2VModel):

    def __init__(self, db_name):
        super().__init__()
        self.vocab = api.load(db_name)
        self.index = {word: idx for idx, word in enumerate(self.vocab.vocab)}

    def __len__(self):
        return len(self.vocab.vocab.keys())

    def word2vec(self, word):
        return self.vocab.get_vector(word)

    def word2index(self, word):
        return self.vocab.vocab[word].index

    def generate_embedding_layer(self, in_len, trainable=False):
        return self.vocab.get_keras_embedding(train_embeddings=trainable)


class BOWModel(W2VModel):
    # Index == Vector

    def __init__(self, w2v_len=32):
        super().__init__()
        self.w2v_len = w2v_len
        self.id = 0
        self.vocab = {}

    def __len__(self):
        return len(self.vocab.keys())

    def word2vec(self, word):
        if word not in self.vocab:
            self.vocab[word] = self.id
            self.id += 1
        return [self.vocab[word]]

    def word2index(self, word):
        return self.word2vec(word)

    def generate_embedding_layer(self, in_len):
        return Embedding(len(self), self.w2v_len, input_length=in_len)
