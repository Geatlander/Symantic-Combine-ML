
from __future__ import print_function
from keras import *
from keras.models import Sequential
from keras.models import load_model

from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import Dropout
from keras import optimizers
import numpy as np
import os

data1 = open("english.txt", "r").read()
data2 = open("french.txt", "r").read()
fulldata = data1 + data2
chars = list(set(fulldata))


VOCAB_SIZE = len(chars)
SEQ_LENGTH = 50
HIDDEN_DIM = 500
LAYER_NUM = 2
BATCH_SIZE = 50
GENERATE_LENGTH = 500




ix_to_char = {ix:char for ix, char in enumerate(chars)}
char_to_ix = {char:ix for ix, char in enumerate(chars)}



class TrainData(object):
    def __init__(self, data, X=None, y=None):
        self.data = data
        self.X = X
        self.y = y
        self.X_arry = None
        self.y_arry = None


    def ret_X(self):
        return self.X

    def ret_y(self):
        return self.y

    def edit_X(self, X):
        self.X = X
    def edit_y(self, y):
        self.y = y

    def split_x_y(self):
        self.X_arry = np.array_split(self.X, 100)
        self.y_arry = np.array_split(self.y, 100)


frenchdata = TrainData(data1)
englishdata = TrainData(data2)

files = [frenchdata, englishdata]
def setupData(datafile):

    datafile.edit_X(np.zeros((len(datafile.data)/SEQ_LENGTH, SEQ_LENGTH, VOCAB_SIZE)))
    datafile.edit_y(np.zeros((len(datafile.data)/SEQ_LENGTH, SEQ_LENGTH, VOCAB_SIZE)))
    for i in range(0, len(datafile.data)/SEQ_LENGTH):
        X_sequence = datafile.data[i*SEQ_LENGTH:(i+1)*SEQ_LENGTH]
        X_sequence_ix = [char_to_ix[value] for value in X_sequence]
        input_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
        for j in range(SEQ_LENGTH):
            input_sequence[j][X_sequence_ix[j]] = 1.
            datafile.X[i] = input_sequence
    
        y_sequence = datafile.data[i*SEQ_LENGTH+1:(i+1)*SEQ_LENGTH+1]
        y_sequence_ix = [char_to_ix[value] for value in y_sequence]
        target_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
        for j in range(SEQ_LENGTH):
            #print("i: " + str(i) + " j: " + str(j))
            target_sequence[j][y_sequence_ix[j]] = 1
            datafile.y[i] = target_sequence

# The actual network creation part
model = Sequential()


model_load = raw_input("Would you like to load a certain model? y/n:  ")
if model_load == "y":
    model_name = raw_input("Enter the file name: ")
    model = load_model(model_name)
else:
    print("A model will now be created from scratch.")
    model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
    for i in range(LAYER_NUM - 1):
        model.add(LSTM(HIDDEN_DIM, return_sequences=True))
        model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(VOCAB_SIZE)))
    #Temperature setting so that the model can learn more liberally
    model.add(Lambda(lambda x: x / 10))
    model.add(Activation('softmax'))
    #Low learning rate set, also to allow less confident learning by the network.
    rms = optimizers.RMSprop(lr=0.00005)
    model.compile(loss="categorical_crossentropy", optimizer=rms)






def generate_text(model, length):
    ix = [np.random.randint(VOCAB_SIZE)]
    y_char = [ix_to_char[ix[-1]]]
    X = np.zeros((1, length, VOCAB_SIZE))
    for i in range(length):
        X[0, i, :][ix[-1]] = 1
        print(ix_to_char[ix[-1]], end="")
        ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        y_char.append(ix_to_char[ix[-1]])
    return ('').join(y_char)



for datafile in files:
    setupData(datafile)
    datafile.split_x_y()
    print(datafile.X.size)
#Training the model, outputting text after each epoch.
nb_epoch = 0
data_index = 0
while True:
    print('\n\n')
    print("Epoch " + str(nb_epoch+1))
    for datafile in files:
        model.fit(datafile.X_arry[data_index], datafile.y_arry[data_index], batch_size=BATCH_SIZE, verbose=1, epochs=1)
    if nb_epoch % 100 == 0: generate_text(model=model, length=GENERATE_LENGTH)
    nb_epoch += 1
    data_index += 1
    if data_index == 100: data_index = 0
    if nb_epoch % 1 == 0:
        #Replace savepath with whatever full directory you want to save the model to:
        savepath = 'home/jason/Desktop/models'
        dirname = os.path.dirname(savepath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            print("Directory created")
        model.save('model.h5')
        print("\nModel saved")

