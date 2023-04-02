from keras.layers import Dense, Embedding, Flatten, Dropout
from keras.models import Sequential
from keras.activations import sigmoid
import pickle
from keras.utils import pad_sequences
import numpy as np

class NameScanner():
    '''Quantity is the number of keywords to be returned.
        Model is the model to be used. It is recommended to try all of
        the models to see which has the right specifications for a given
        task.
        Bothgrams, when set to true, returns keywords of lengths
        one and two. When false, only single-word keywords are
        returned.'''
    def __init__(self, quantity=None, model=1, bothgrams=False):
        self.bothgrams = bothgrams
        self.quantity = quantity
        if model == 1:
            self.maxlen = 4
            self.model = Sequential()
            self.model.add(Embedding(10000, 16, input_length=self.maxlen))
            self.model.add(Flatten())
            self.model.add(Dense(64))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(1, activation=sigmoid))
            self.model.compile(optimizer="adam", loss="binary_crossentropy")
            self.model.fit([[0,0,0,0]], [0], epochs=1, verbose=False)
            self.model.load_weights("models/model1/weights1")
            with open('models/model1/tokenizer1.pickle', 'rb') as f:
                self.tokenizer = pickle.load(f)
        if model == 2:
            self.maxlen = 4
            self.model = Sequential()
            self.model.add(Embedding(10000, 8, input_length=self.maxlen))
            self.model.add(Flatten())
            self.model.add(Dense(128))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(16))
            self.model.add(Dropout(0.2))
            self.model.add(Dense(1, activation=sigmoid))
            self.model.compile(optimizer="adam", loss="binary_crossentropy")
            self.model.fit([[0,0,0,0]], [0], epochs=1, verbose=False)
            self.model.load_weights("models/model2/weights2")
            with open('models/model2/tokenizer2.pickle', 'rb') as f:
                self.tokenizer = pickle.load(f)

    def is_name(self, phrase):
        sequences = pad_sequences(self.tokenizer.texts_to_sequences([phrase]), maxlen=self.maxlen)
        return self.model.predict(np.array(sequences), verbose=False)[0]

    def find_names(self, text):
        text = text.split()
        text = list(filter(lambda x: str.isalpha(x), text))
        twograms = []
        for i in range(len(text)-1):
            if not(text[i]).islower() and not(text[i+1]).islower():
                twograms.append(text[i]+" "+text[i+1])
        onegrams = list(filter(lambda x: not str.islower(x), text))
        if self.bothgrams == True:
            words = twograms + onegrams
        else:
            words = onegrams
        indicator = [self.is_name(word) for word in words]
        out = []
        for i in range(len(words)):
            if np.round(indicator[i]):
                out.append(words[i])
        if self.quantity == None:
            return set(out)
        else:
            return set(out[:self.quantity])