import sys, os, os.path, glob
import numpy as np
import pandas as pd
import pickle
import datetime
import argparse

import matplotlib.pyplot as plt

import sklearn.linear_model
import sklearn.model_selection
import sklearn.metrics # mean_squared_error, mean_absolute_percentage_error, r2_score


parser = argparse.ArgumentParser(description='input parameters')
parser.add_argument('--csvpath', type=str, default='cities-forecasting', help='data path of csv files')
parser.add_argument('--datecol', type=str, default='Date')
parser.add_argument('--targetcol', type=str, default='PM2.5')
parser.add_argument('--lastmcol', type=str, default='LastDayPM2.5')
parser.add_argument('--savepath', type=str, default='cities-lr', help='save path for linear regression predictions')

args = parser.parse_args()

# create dir
os.makedirs(args.savepath, exist_ok=True)

# get all city files from dir
filelist = glob.glob('{}/*.csv'.format(args.csvpath))
filelist.sort()

# record evaluation metrics for each city
mae_list = [] 
mape_list = []
r2_list = []

for fn in filelist:
    # load city csv file and set index
    basefn = os.path.basename(fn)
    print(basefn)
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
    print('all_cols', all_cols)
    print('feature_cols', feature_cols)
    
    # drop missing data and get x, y
    dfci_data = dfci.get([target]+feature_cols)
    dfci_data = dfci_data.dropna()
    X = dfci_data.get(feature_cols).values
    y = dfci_data[target].values
    print('input X', X.shape)
    if X.shape[0] < 10:
        # do not predict if there are less than 10 data points
        continue
    
    # split into 80% training and 20% testing
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.5)
    ns_train = x_train.shape[0]
    print('x_train', x_train.shape)
    print('x_test', x_test.shape, X.shape[0]-x_train.shape[0])

    # train the model
    linear = sklearn.linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    
    # perform prediction
    y_pred = linear.predict(x_test) 
    
    # evaluate
    mae  = sklearn.metrics.mean_absolute_error(y_test, y_pred)
    mape = sklearn.metrics.mean_absolute_percentage_error(y_test, y_pred)
    r2   = sklearn.metrics.r2_score(y_test, y_pred)
    print(mae, mape, r2)
    
    # global evaluation
    mae_list.append(mae)
    mape_list.append(mape)
    r2_list.append(r2)

    dfpred = pd.DataFrame({
        args.datecol: dfci_data.index[ns_train:],
        'truth': y_test,
        'pred': y_pred,
    })
    dfpred.to_csv('{}/{}'.format(args.savepath, basefn), index=False)

    
print('mean MAE', np.mean(mae_list))
print('mean MAPE', np.mean(mape_list))
print('mean R2', np.mean(r2_list))