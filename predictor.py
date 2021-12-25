import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import numpy as np

import datetime
import os
import json

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

from sklearn.metrics import mean_squared_error

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Prepare Data


rawPrice = pd.read_csv('./data/price.csv')
priceTitle = [d[0:4] + '-' + d[4:6] + '-' + d[6:] for d in rawPrice.columns[1:]]
priceTitle.insert(0, 'Date')
rawPrice.columns = priceTitle
rawPrice.set_index(['Date'], inplace=True)
rawPrice = rawPrice.T
rawPrice.to_csv('./processed_data/processed_price.csv')

rawPrice.index = pd.to_datetime(rawPrice.index, format="%Y-%m-%d")
print(rawPrice.shape)
rawPrice.head()


cluster = pd.read_csv('./processed_data/cluster.csv')
cluster.columns = ['ts_code', 'cluster_id']
cluster = cluster.sort_values(by='cluster_id')
cluster.to_csv('./processed_data/processed_cluster.csv')

print(cluster.shape)
cluster.head()



def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)



# Train


def train(m, index, epo, verbose=False):
    stock = rawPrice[[index]]
    stock = stock.dropna()
    if verbose:
        sns.set()
        plt.figure(figsize=(20, 10))
        plt.plot(stock, color="black")
        plt.ylabel("Price")
        plt.xlabel("Date")
        plt.xticks(rotation=45)
        plt.show()

    X, y = split_sequence(stock.T.values[0], 20)
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    X_train = X[0:int(len(X) * 0.8)]
    X_test = X[int(len(X) * 0.8):]
    y_train = y[0:int(len(X) * 0.8)]
    y_test = y[int(len(X) * 0.8):]
    print('Train sample Num: ', len(X_train), ', Test sample Num: ', len(X_test))

    m.fit(X_train, y_train, epochs=epo, verbose=0)

    res = [m.predict(X_test[i].reshape((1, n_steps, n_features)), verbose=0)[0][0] for i in range(len(X_test))]
    mse_ = mean_squared_error(res, y_test)
    mse_dict[idx][1] = mse_
    print('MSE for stock{}: '.format(idx), mean_squared_error(res, y_test))



def train_cluster(m, clusterIndex, epo, verbose=False):
    X_all = []
    y_all = []
    X_test_split = {}
    y_test_split = {}
    for idx in cluster[cluster['cluster_id'] == clusterIndex]['ts_code'].values:
        stock = rawPrice[[idx]]
        stock = stock.dropna()
        X, y = split_sequence(stock.T.values[0], 20)
        X = X.reshape((X.shape[0], X.shape[1], n_features))
        X_test_split[idx] = X[int(len(X) * 0.8):]
        y_test_split[idx] = y[int(len(X) * 0.8):]

        for i in range(X.shape[0]):
            X_all.append(X[i])
            y_all.append(y[i])

    X_train = X_all[0:int(len(X_all) * 0.8)]
    X_test = X_all[int(len(X_all) * 0.8):]
    y_train = y_all[0:int(len(y_all) * 0.8)]
    y_test = y_all[int(len(y_all) * 0.8):]
    print('Train sample Num: ', len(X_train), ', Test sample Num: ', len(X_test))

    m.fit(np.array(X_train), np.array(y_train), epochs=epo, verbose=verbose)

    for idx in X_test_split.keys():
        res = [m.predict(X_test_split[idx][i].reshape((1, n_steps, n_features)), verbose=0)[0][0] for i in
               range(len(X_test_split[idx]))]
        mse_ = mean_squared_error(res, y_test_split[idx])
        mse_dict[idx] = [mse_, 0]
        print('MSE for stock{}: '.format(idx), mean_squared_error(res, y_test_split[idx]))



# Main


n_steps = 20
n_features = 1
mse_dict = {}


# Our method
our_models = list()
for clu_id in list(set(cluster['cluster_id'].values)):
    model = Sequential()
    model.add(LSTM(150, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(150, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    our_models.append(model)

    print('\nStocks in cluster{}: '.format(clu_id), cluster[cluster['cluster_id'] == clu_id]['ts_code'].values)
    train_cluster(model, clu_id, 200, verbose=False)


# common_models = list()
# for idx in cluster[cluster['cluster_id'] == 0]['ts_code'].values:
#     model = Sequential()
#     model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
#     model.add(LSTM(50, activation='relu'))
#     model.add(Dense(1))
#     model.compile(optimizer='adam', loss='mse')

#     print('='*20+'Stock '+str(idx)+'='*20)
#     train(model, idx, 20)
#     common_models.append(model)


# Baseline
our_models = list()
for idx in rawPrice.columns.to_list():
    if idx == 563:
        continue
    model = Sequential()
    model.add(LSTM(150, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(LSTM(150, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    our_models.append(model)

    train(model, idx, 200)


for idx in mse_dict.keys():
    mse_dict[idx].append(mse_dict[idx][0] - mse_dict[idx][1])
    print('MSE Improvement for stock {} is {}'.format(idx, mse_dict[idx][2]))


res = {}
for k in mse_dict.keys():
    res[str(k)] = mse_dict[k]

jsObj = json.dumps(res)
fileObject = open('./processed_data/res.json', 'w')
fileObject.write(jsObj)
fileObject.close()
