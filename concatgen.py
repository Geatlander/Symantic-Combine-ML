
from __future__ import print_function
from keras import *
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Activation
import numpy as np
import os
import random

data1 = open("english.txt", "r").readlines()
data2 = open("french.txt", "r").readlines()
data = data1 + data2
random.shuffle(data)
data = str(data)
#data = filter(lambda x: x != '\\n', data)
#data = filter(lambda x: x != "\\xc3\\xa", data)
#data = filter(lambda x: x != "\\xc2\\xa", data)
#data = filter(lambda x: x == "\n", data)
#print(data)

chars = list(set(data))
VOCAB_SIZE = len(chars)
SEQ_LENGTH = 50
HIDDEN_DIM = 500
LAYER_NUM = 2
BATCH_SIZE = 50
GENERATE_LENGTH = 1000

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



ix_to_char = {ix:char for ix, char in enumerate(chars)}
char_to_ix = {char:ix for ix, char in enumerate(chars)}

X = np.zeros((len(data)/SEQ_LENGTH, SEQ_LENGTH, VOCAB_SIZE))
y = np.zeros((len(data)/SEQ_LENGTH, SEQ_LENGTH, VOCAB_SIZE))
for i in range(0, len(data)/SEQ_LENGTH):
    X_sequence = data[i*SEQ_LENGTH:(i+1)*SEQ_LENGTH]
    X_sequence_ix = [char_to_ix[value] for value in X_sequence]
    input_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
    for j in range(SEQ_LENGTH):
        input_sequence[j][X_sequence_ix[j]] = 1.
        X[i] = input_sequence
    
    y_sequence = data[i*SEQ_LENGTH+1:(i+1)*SEQ_LENGTH+1]
    y_sequence_ix = [char_to_ix[value] for value in y_sequence]
    target_sequence = np.zeros((SEQ_LENGTH, VOCAB_SIZE))
    for j in range(SEQ_LENGTH):
        target_sequence[j][y_sequence_ix[j]] = 1
        y[i] = target_sequence

# The actual network creation part
model = Sequential()


model_load = raw_input("Would you like to load a certain model? y/n:  ")
if model_load == "y":
    model_name = raw_input("Enter the file name: ")
    model = load_model(model_name)
    output_text = raw_input("Would you like to immediately output text? y/n: ")
    if output_text == "y":
        generate_text(model, 100000)
else:
    print("A model will now be created from scratch.")
    model.add(LSTM(HIDDEN_DIM, input_shape=(None, VOCAB_SIZE), return_sequences=True))
    for i in range(LAYER_NUM - 1):
        model.add(LSTM(HIDDEN_DIM, return_sequences=True))
    model.add(TimeDistributed(Dense(VOCAB_SIZE)))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")




#Training
nb_epoch = 0
while True:
    print('\n\n')
    print("Epoch " + str(nb_epoch+1))
    model.fit(X, y, batch_size=BATCH_SIZE, verbose=1, epochs=1)
    nb_epoch += 1
    generate_text(model=model, length=GENERATE_LENGTH)
    if nb_epoch % 10 == 0:
        #Replace savepath with whatever full directory you want to save the model to:
        savepath = 'home/jason/Desktop/models'
        dirname = os.path.dirname(savepath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            print("Directory created")
        model.save('model.h5')
        print("Model saved")

