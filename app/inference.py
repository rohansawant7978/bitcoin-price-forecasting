import quandl
import investpy
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import numpy as np
import pandas_ta as ta
from datetime import datetime,timedelta
from sklearn.preprocessing import RobustScaler,MinMaxScaler
import pickle

def pred():
    soup = str(requests.get('https://bitinfocharts.com/comparison/transactionvalue-btc.html').content)
    scraped_output = (soup.split('[[')[1]).split('{labels')[0][0:-2]
    date_value_array = scraped_output.replace('new Date(','').replace(')','').replace('[','').replace(']','').replace('"','').replace('/','-').split(',')
    data_dict = dict(zip(date_value_array[::2],date_value_array[1::2]))
    avg_transaction_value = data_dict[(datetime.today()-timedelta(days=1)).strftime('%Y-%m-%d')]
    avg_tr_value_df = pd.DataFrame()
    avg_tr_value_df['Date'] = pd.to_datetime(date_value_array[::2])
    avg_tr_value_df['avg_transaction_value'] = pd.to_numeric(date_value_array[1::2])

    yesterday_date = (datetime.today()-timedelta(days=1)).strftime('%d/%m/%Y')
    ohlc_df = investpy.get_crypto_historical_data(crypto='bitcoin',from_date='01/01/2013',to_date=yesterday_date)
    ohlc_df = ohlc_df.reset_index().drop(['Currency','Volume','High','Low'],axis=1)
    ohlc_df.columns = ['Date','opening_price','closing_price']

    final_df = pd.merge(avg_tr_value_df,ohlc_df,on=['Date'],how='inner')

    close_bband_df_30 = ta.bbands(final_df['closing_price'],30)
    final_df['bband_upper30 closing_price']  = close_bband_df_30[f'BBU_{30}_2.0'] 
    final_df['tema7 closing_price']  = ta.tema(final_df['closing_price'],7) 

    final_df['dema7 opening_price']  = ta.dema(final_df['opening_price'],7)  

    final_df['ema30 avg_transaction_value']  = ta.ema(final_df['avg_transaction_value'],30)
    final_df['ema90 avg_transaction_value']  = ta.ema(final_df['avg_transaction_value'],90)
    final_df['sma90 avg_transaction_value']  = ta.sma(final_df['avg_transaction_value'],90)
    final_df['tema90 avg_transaction_value']  = ta.tema(final_df['avg_transaction_value'],90)
    final_df['wma30 avg_transaction_value']  = ta.wma(final_df['avg_transaction_value'],30)

    final_df = final_df[(final_df['Date'] >= '2013-04-01')]

    final_df = final_df[['Date', 'sma90 avg_transaction_value', 'ema90 avg_transaction_value',
                        'dema7 opening_price', 'closing_price', 'wma30 avg_transaction_value',
                        'ema30 avg_transaction_value', 'tema90 avg_transaction_value',
                        'opening_price', 'tema7 closing_price', 'bband_upper30 closing_price']]

    sgd_reg = pickle.load(open('linear_reg_10_25.sav', 'rb'))

    X = final_df.drop(['Date'],axis=1)

    scaler = RobustScaler()
    X_scaled = X.copy()
    X_scaled[X.columns] = scaler.fit_transform(X[X.columns])

    scaler = MinMaxScaler()
    X_scaled[X.columns] =  scaler.fit_transform(X_scaled[X.columns])

    today_btc_closing_price =  sgd_reg.predict(X_scaled.values[-1].reshape(1,-1))
    final_df['Date'] = final_df['Date'].dt.strftime('%d-%m-%Y')
    return final_df,round(float(today_btc_closing_price))

def get_current_close_price():
    return investpy.get_crypto_historical_data(crypto='bitcoin',from_date='01/01/2021',to_date=datetime.today().strftime('%d/%m/%Y'))['Close'][-1]

def get_df():
    return investpy.get_crypto_historical_data(crypto='bitcoin',from_date='01/04/2013',to_date=datetime.today().strftime('%d/%m/%Y')).reset_index()[['Date','Close']]
