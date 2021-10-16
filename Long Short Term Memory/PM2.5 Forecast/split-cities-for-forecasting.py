import sys, os, os.path
import numpy as np
import pandas as pd
import pickle
import datetime
import argparse

parser = argparse.ArgumentParser(description='input parameters')
parser.add_argument('--sourcecsv', type=str, default='city_day.csv', help='original csv files')
parser.add_argument('--citycol', type=str, default='City')
parser.add_argument('--datecol', type=str, default='Date')
parser.add_argument('--targetcol', type=str, default='PM2.5')
parser.add_argument('--nextmcol', type=str, default='NextDayPM2.5')
parser.add_argument('--removecols', type=str, default='City,AQI,AQI_Bucket,')
parser.add_argument('--savepath', type=str, default='cities-forecasting', help='save path for csv files')


def reindex_time_series(df):
    start_day, end_day = min(df.index), max(df.index)
    timeindex = pd.date_range(start=start_day, end=end_day, freq='D')
    df = df.reindex(timeindex)
    return df


def add_last_measurement(df, targetcol, nextmcol):
    ns, nc = df.shape
    nextd_m = df[targetcol].tolist()[1:] + [np.nan]
    df[nextmcol] = nextd_m
    return df


def remove_extra_cols(df, removecols, targetcol, nextd_m):
    removecols = [i for i in removecols if len(i) > 0]
    cols = list(df)
    for i in [nextd_m,targetcol]+removecols:
        if i in cols:
            cols.remove(i)

    cols.sort()
    df = df.get([nextd_m,targetcol]+cols)
    return df


# parse input parameters
args = parser.parse_args()

# loading the data
data = pd.read_csv(args.sourcecsv)
data[args.datecol] = pd.to_datetime(data[args.datecol])

# get all unique cities
city_list = data[args.citycol].unique().tolist()

# create dir
os.makedirs(args.savepath, exist_ok=True)

# separate data into different cities and add next day pm2.5 
df_list = []
sum1= 0
sum2= 0
for city in city_list:
    dfci = data.loc[data[args.citycol] == city]
    dfci = dfci.set_index(args.datecol)
    dfci = dfci.sort_index()
    
    dfci = reindex_time_series(dfci)
    dfci.index.rename(args.datecol, inplace=True)
    
    dfci = add_last_measurement(dfci, args.targetcol, args.nextmcol)
    dfci = remove_extra_cols(dfci, args.removecols.split(','), args.targetcol, args.nextmcol)

    df_list.append(dfci)
    dfci.to_csv('{}/{}.csv'.format(args.savepath, city))