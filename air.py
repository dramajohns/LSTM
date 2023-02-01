# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 09:50:23 2022

@author: fedib
"""
import pandas as pd 
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras import layers,Sequential
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding,Bidirectional
from tensorflow.keras.optimizers import RMSprop,Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
dataset = pd.read_csv('C:/STUDY/CIII/Deep learning/tp/tp5 air/air.csv')
x=np.array(dataset.drop(columns="Month"))
length=len(x)

scaler = MinMaxScaler(feature_range=(0,1))
x=scaler.fit_transform(x)


label=np.array([x[i] for i in range(1,len(x))])
data=np.array([x[i] for i in range(0,len(x)-1)])
data1=data.reshape([data.shape[0],1,data.shape[1]])

#train_data,test_data,train_label,test_label=train_test_split(datanew,label,test_size = 0.2,random_state = 0)

for i in

model=Sequential()
model.add(LSTM(4,input_shape=(1,1)))
model.add(Dense(4,activation='sigmoid'))
model.compile('adam',loss='mse',metrics=['mse'])
result=model.fit(train_data,train_data,epochs=100,validation_data=(test_data,test_label))


model.summary()
