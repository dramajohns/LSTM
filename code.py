# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 11:20:57 2022

@author: fedib
"""
import string
from keras.preprocessing.text import Tokenizer
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras import layers,Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding,Bidirectional
from tensorflow.keras.optimizers import RMSprop,Adam

text="""Artificial Intelligence (AI) is a technical
science that studies and develops theories,
methods, technologies, and applications for
simulating and extending human intelligence. The
purpose of AI is to enable machines to think like
people and to make machines intelligent. Today,
AI has become an interdisciplinary course that
involves various fields."""



tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

sequences=tokenizer.texts_to_sequences([text])[0]
len(sequences)

x=[sequences[i] for i in range(0,len(sequences)-1)]
y=[sequences[i] for i in range(1,len(sequences))]
max_words=40
x=np.array(x)
y=np.array(y)
y=to_categorical(y, num_classes=max_words)



model=Sequential()
model.add(Embedding(40,40,input_length=1))
model.add(LSTM(50))
#model.add(Dense(max_words,activation='relu'))
model.add(Dense(40,activation='softmax'))
#adam = Adam(lr=0.01)
model.compile('adam',loss='categorical_crossentropy',metrics=['accuracy'])
result=model.fit(x,y,epochs=100)


#model.predict(8)

#sequences=tokenizer.texts_to_sequences([text])[0]
#sequence=np.array(sequences)

word_convert=tokenizer.texts_to_sequences(["various"])[0]

word_index=np.argmax(model.predict(word_convert))

for key, value in tokenizer.word_index.items():
            if value ==word_index:
                predicted_word = key
                break
        
print(predicted_word)   

