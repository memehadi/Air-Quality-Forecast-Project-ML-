import sys, os, os.path, glob
import pickle
import datetime
import argparse
import numpy as np
import pandas as pd
import sklearn

import matplotlib.pyplot as plt
from matplotlib import style

from sklearn import linear_model
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, BatchNormalization, multiply, concatenate, Flatten, Activation, dot
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


parser = argparse.ArgumentParser(description='input parameters')
parser.add_argument('--csvpath', type=str, default='cities-original', help='data path of csv files')
parser.add_argument('--citycol', type=str, default='City')
parser.add_argument('--datecol', type=str, default='Date')
parser.add_argument('--targetcol', type=str, default='PM2.5')
parser.add_argument('--inseq', type=int, default=60)
parser.add_argument('--outseq', type=int, default=30)
parser.add_argument('--savepath', type=str, default='cities-seq2seq', help='save path for predictions')
parser.add_argument('--modelpath', type=str, default='savedmodels', help='save path for models')

args = parser.parse_args()

# create dir
os.makedirs(args.savepath, exist_ok=True)


def normalize(df, features):
    min_max_v = []

    for fi in features:
        v = df[fi]
        v = v.dropna()
        min_max_v.append((v.min(), v.max()))
 
    for fi, mm in zip(features, min_max_v):
        minv, maxv = mm
        vv = df[fi].values
        vn = (vv - minv) / (maxv - minv)
        df[fi] = vn
    
    return df, features, min_max_v


def inverse_normalize(matrix, indices, features, min_max_v):
    df = pd.DataFrame(data=matrix, index=indices, columns=features)

    for fi, mm in zip(features, min_max_v):
        minv, maxv = mm
        vv = df[fi].values
        vn = (vv * (maxv - minv)) + minv
        df[fi] = vn
    
    return df



def truncate(x, feature_cols=range(4), target_cols=range(4), train_len=100, test_len=20):
    in_, out_, idl = [], [], []
    for i in range(len(x)-train_len-test_len+1):
        xi = x[i:(i+train_len), feature_cols].tolist()
        yi = x[(i+train_len):(i+train_len+test_len), target_cols].tolist()
        if np.isnan(xi).any() or np.isnan(yi).any():
            # do not use data because of nan values
            pass
        else:
            in_.append(xi)
            out_.append(yi)
            idl.append(i+train_len)
    return np.array(in_), np.array(out_), idl


def create_model(inputshape, outputshape, n_hidden):
    input_neurons = Input(shape=inputshape)
    output_neurons = Input(shape=outputshape)

    # encoder
    encoder_last_h1, encoder_last_h2, encoder_last_c = LSTM(n_hidden, activation='relu', dropout=0.2, recurrent_dropout=0.2, return_sequences=False, return_state=True)(input_neurons)
    print(encoder_last_h1)
    print(encoder_last_h2)
    print(encoder_last_c)

    encoder_last_h1 = BatchNormalization(momentum=0.6)(encoder_last_h1)
    encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)

    # decoder
    decoder = RepeatVector(output_neurons.shape[1])(encoder_last_h1)
    decoder = LSTM(n_hidden, activation='relu', dropout=0.2, recurrent_dropout=0.2, return_state=False, return_sequences=True)(decoder, initial_state=[encoder_last_h1, encoder_last_c])
    print(decoder)

    decoder = RepeatVector(output_neurons.shape[1])(encoder_last_h1)
    decoder = LSTM(n_hidden, activation='relu', dropout=0.2, recurrent_dropout=0.2, return_state=False, return_sequences=True)(decoder, initial_state=[encoder_last_h1, encoder_last_c])
    print(decoder)

    # time distributed
    out = TimeDistributed(Dense(output_neurons.shape[2]))(decoder)
    print(out)

    model = Model(inputs=input_neurons, outputs=out)
    opt = Adam(lr=0.01, clipnorm=1)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
    model.summary()

    return model


# create dir
os.makedirs(args.savepath, exist_ok=True)

# get all city files from dir
filelist = glob.glob('{}/*.csv'.format(args.csvpath))
filelist.sort()

# get train and test data
# x_train = []
# y_train = []

# x_test = []
# y_test = []

# test_size = []

for fn in filelist:
    # load city csv file and set index
    basefn = os.path.basename(fn)
    print(basefn)
    city = basefn.replace('.csv', '')
    dfci = pd.read_csv(fn)
    dfci[args.datecol] = pd.to_datetime(dfci[args.datecol])
    dfci = dfci.set_index(args.datecol)
    dfci = dfci.sort_index()
    
    # get feature cols
    all_cols = list(dfci)
    target = args.targetcol
    feature_cols = list(dfci)
    feature_cols.remove(target)
    
    # remove features where missing data is more than 90%
    remove_features = []
    for feature in feature_cols:
        dffi = dfci.loc[:, feature]
        ns_full = dffi.shape[0]
        dffid = dffi.dropna()
        ns_data = dffid.shape[0]
        missing_data_percentage = 1.0 - (ns_data/ns_full)
        print('for feature {}, data points {}/{}, missing data percentage is {}'.format(feature, ns_data,ns_full, missing_data_percentage))
        if missing_data_percentage > 0.9:
            remove_features.append(feature)
    
    for fi in remove_features:
        feature_cols.remove(fi)
    feature_cols.sort()
    print('all_cols', all_cols)
    print('feature_cols', feature_cols)
    
    # drop missing data and get x, y
    dfci_data = dfci.get([target]+feature_cols)
    dfci_nrm, all_features, min_max_v = normalize(dfci_data.copy(), [target]+feature_cols)
    nfeatures = len(all_features)
    X_ci, Y_ci, pindices = truncate(dfci_nrm.get([target]+feature_cols).values, feature_cols=range(nfeatures), target_cols=range(nfeatures), train_len=args.inseq, test_len=args.outseq)

    if X_ci.shape[0] < 10:
        # do not predict if there are less than 10 data points
        continue
    
    print(pindices)
    # split into 50% training and testing
    x_train_ci, x_test_ci, y_train_ci, y_test_ci = sklearn.model_selection.train_test_split(X_ci, Y_ci, test_size=0.5)
    ns_train = x_train_ci.shape[0]
    print('x_train_ci', x_train_ci.shape)
    print('x_test_ci', x_test_ci.shape, X_ci.shape[0]-x_train_ci.shape[0])
    
    #test_size.append((city, x_test_ci.shape[0]))

    #x_train.append(x_train_ci)
    #y_train.append(y_train_ci)

    #x_test.append(x_test_ci)
    #y_test.append(y_test_ci)

    print('--------train test data shape')
    print(x_train_ci.shape)
    print(y_train_ci.shape)
    print(x_test_ci.shape)
    print(y_test_ci.shape)
    print('--------end of train test data shape')

    # create model
    model = create_model(x_train_ci.shape[1:], y_train_ci.shape[1:], 100)

    # train the model
    epc = 100
    es = EarlyStopping(monitor='val_loss', mode='min', patience=50)
    history = model.fit(x_train_ci, y_train_ci, validation_split=0.2, epochs=epc, verbose=1, callbacks=[es], batch_size=100)
    train_mae = history.history['mae']
    valid_mae = history.history['val_mae']

    model.save('{}/model_forecasting_seq2seq_{}.h5'.format(args.modelpath, city))

    # prediction
    test_pred_ci = model.predict(x_test_ci)
    ns, nt, nf = test_pred_ci.shape
    
    pprd_indices = np.array(pindices[ns_train:])
    print(pprd_indices)
    for ti in range(nt):
        test_mi = test_pred_ci[:, ti, :]
        timeindex = dfci_nrm.iloc[pprd_indices + ti, :].index
        df_pred_ci_ti = inverse_normalize(test_mi, timeindex, [target]+feature_cols, min_max_v)
        df_true_ci_ti = dfci_data.loc[timeindex, [target]+feature_cols]

        df_pred_ci_ti.to_csv('{}/{}-{}-pred.csv'.format(args.savepath, city, ti), index=True)
        df_true_ci_ti.to_csv('{}/{}-{}-true.csv'.format(args.savepath, city, ti), index=True)
    
  

# record evaluation metrics for each city



#x_train = np.concatenate(x_train, axis=0)
#y_train = np.concatenate(y_train, axis=0)

#x_test = np.concatenate(x_test, axis=0)
#y_test = np.concatenate(y_test, axis=0)