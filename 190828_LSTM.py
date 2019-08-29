# -*- coding: utf-8 -*-
import random
time=[]
def create_timeserise():
    #2019-6-1
    start_time = 1563871130
    for i in range(30):
        #1~7 days as a range to change the price
        start_time = start_time + random.randint(60*60*24,60*60*24*14)
        time.append(start_time)
create_timeserise()
#print(time) 

inteval = []
sum_inteval = 0
for j in range(29):
    inteval_time = time[29-j]-time[28-j]
    inteval.append(inteval_time)
    sum_inteval = sum_inteval + inteval_time
    
#print(inteval,sum_inteval)
print(round(sum_inteval/(30*60*60*24)))  

"""    
print("\n")  
for j in range(29):   
    print(random.randint(86400,604800))  
"""
price = []
def create_priceserise():
    for i in range(30):
        min_price = random.uniform(20,50)
        max_price = random.uniform(120,300)
        actual = max_price - min_price
        price.append(actual)

create_priceserise()
#print(price)

import pandas as pd
df = pd.DataFrame(list(zip(time,price)),columns = ['Time','Price'])
#print(df)

import matplotlib.pyplot as plt
def train_test_split(df, test_size=0.1):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data
    
    
def line_plot(train_x,test_x,line1, line2, label1 = None, label2 = None, title=''):
    #plt.figure()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.plot(train_x,line1, label=label1, linewidth=2)#draw first line
    ax.plot(test_x,line2, label=label2, linewidth=2)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_xlabel('time', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=18)

train, test = train_test_split(df, test_size=0.1)
line_plot(train.Time, test.Time, train.Price, test.Price, 'training', 'test', '')

import numpy as np
def normalise_zero_base(df):
    """ Normalise dataframe column-wise to reflect changes with respect to first entry."""
    return df / df.iloc[0] - 1
def extract_window_data(df, window=7, zero_base=True):
    """ Convert dataframe to overlapping sequences/windows of length `window`."""
    window_data = []
    for idx in range(len(df) - window):
        tmp = df[idx: (idx + window)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)
def prepare_data(df, target_col, window=7, zero_base=True, test_size=0.2):
    """ Prepare data for LSTM. """
    # train test split
    train_data, test_data = train_test_split(df, test_size)
    
    # extract window data
    X_train = extract_window_data(train_data, window, zero_base)
    X_test = extract_window_data(test_data, window, zero_base)
    
    # extract targets
    y_train = train_data.Price[window:].values
    y_test = test_data.Price[window:].values
    if zero_base:
        y_train = y_train / train_data.Price[:-window].values - 1
        y_test = y_test / test_data.Price[:-window].values - 1    
        return train_data, test_data, X_train, X_test, y_train, y_test

def build_lstm_model(input_data, output_size, neurons=20,
                     activ_func='linear', dropout=0.25,
                     loss='mae', optimizer='adam'):
    model = Sequential()  
    
    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))    
    model.compile(loss=loss, optimizer=optimizer)
    return model


np.random.seed(42)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import Activation
# data params
window = 7
test_size = 0.1
zero_base = True
target_col = 'Price'
# model params
lstm_neurons = 20
epochs = 50
batch_size = 4
loss = 'mae'
dropout = 0.25
optimizer = 'adam'

train, test, X_train, X_test, y_train, y_test = prepare_data(
    df, target_col, window=window, zero_base=zero_base, test_size=test_size)

model = build_lstm_model(
    X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
    optimizer=optimizer)
history = model.fit(
    X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)