
class BOWModel:
    id = 0

    def __init__(self):
        self.vocab = {}

    def __getitem__(self, key):
        if key not in self.vocab:
            self.vocab[key] = BOWModel.id
            BOWModel.id += 1
        return self.vocab[key]

    def __len__(self):
        return len(self.vocab.keys())