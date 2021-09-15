import quandl
import investpy
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import numpy as np
import pandas_ta as ta
from datetime import datetime,timedelta
from functools import reduce
from sklearn.preprocessing import RobustScaler,MinMaxScaler
import pickle

def final_func_1():
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
    ohlc_df = ohlc_df.reset_index().drop(['Currency','Volume'],axis=1)
    ohlc_df.columns = ['Date','opening_price','highest_price','lowest_price','closing_price']

    miners_revenue_df = quandl.get("BCHAIN/MIREV",authtoken='9ztFCcK4_e1xGo_gjzK7')
    miners_revenue_df = miners_revenue_df.rename(columns={'Value': 'miner_revenue'})

    btc_in_circulation_df = quandl.get("BCHAIN/TOTBC",authtoken='9ztFCcK4_e1xGo_gjzK7')
    btc_in_circulation_df = btc_in_circulation_df.rename(columns={'Value': 'number_of_coins_in_circulation'})

    data_frames = [avg_tr_value_df,ohlc_df,miners_revenue_df,btc_in_circulation_df]
    final_df = reduce(lambda  left,right: pd.merge(left,right,on=['Date'],how='inner'),data_frames)

    close_bband_df_30 = ta.bbands(final_df['closing_price'],30)
    close_bband_df_90 = ta.bbands(final_df['closing_price'],90)
    final_df['bband_upper30 closing_price']  = close_bband_df_30[f'BBU_{30}_2.0'] 
    final_df['bband_upper90 closing_price']  = close_bband_df_90[f'BBU_{90}_2.0']  
    final_df['dema7 closing_price']  = ta.dema(final_df['closing_price'],7) 
    final_df['dema30 closing_price']  = ta.dema(final_df['closing_price'],30) 
    final_df['dema90 closing_price']  = ta.dema(final_df['closing_price'],90) 
    final_df['ema90 closing_price']  = ta.ema(final_df['closing_price'],90) 
    final_df['sma90 closing_price']  = ta.sma(final_df['closing_price'],90) 
    final_df['tema7 closing_price']  = ta.tema(final_df['closing_price'],7) 
    final_df['tema30 closing_price']  = ta.tema(final_df['closing_price'],30) 
    final_df['tema90 closing_price']  = ta.tema(final_df['closing_price'],90) 

    high_bband_df_7 = ta.bbands(final_df['highest_price'],7)
    high_bband_df_30 = ta.bbands(final_df['highest_price'],30)
    high_bband_df_90 = ta.bbands(final_df['highest_price'],90)
    final_df['bband_upper7 highest_price']  = high_bband_df_7[f'BBU_{7}_2.0'] 
    final_df['bband_upper30 highest_price']  = high_bband_df_30[f'BBU_{30}_2.0'] 
    final_df['bband_upper90 highest_price']  = high_bband_df_90[f'BBU_{90}_2.0']  
    final_df['dema30 highest_price']  = ta.dema(final_df['highest_price'],30) 
    final_df['dema90 highest_price']  = ta.dema(final_df['highest_price'],90) 
    final_df['ema90 highest_price']  = ta.ema(final_df['highest_price'],90)
    final_df['sma30 highest_price']  = ta.sma(final_df['highest_price'],30) 
    final_df['sma90 highest_price']  = ta.sma(final_df['highest_price'],90) 
    final_df['tema7 highest_price']  = ta.tema(final_df['highest_price'],7) 
    final_df['tema30 highest_price']  = ta.tema(final_df['highest_price'],30) 
    final_df['tema90 highest_price']  = ta.tema(final_df['highest_price'],90) 
    final_df['wma30 highest_price']  = ta.wma(final_df['highest_price'],30) 

    open_bband_df_7 = ta.bbands(final_df['opening_price'],7)
    open_bband_df_90 = ta.bbands(final_df['opening_price'],90)
    final_df['bband_lower7 opening_price']  = open_bband_df_7[f'BBL_{7}_2.0'] 
    final_df['bband_upper90 opening_price']  = open_bband_df_90[f'BBU_{90}_2.0']
    final_df['dema7 opening_price']  = ta.dema(final_df['opening_price'],7)  
    final_df['dema30 opening_price']  = ta.dema(final_df['opening_price'],30)  
    final_df['ema90 opening_price']  = ta.ema(final_df['opening_price'],90)
    final_df['sma90 opening_price']  = ta.sma(final_df['opening_price'],90) 
    final_df['tema30 opening_price']  = ta.tema(final_df['opening_price'],30) 

    final_df['sma7 number_of_coins_in_circulation']  = ta.sma(final_df['number_of_coins_in_circulation'],7) 

    low_bband_df_90 = ta.bbands(final_df['lowest_price'],90)
    final_df['bband_upper90 lowest_price']  = low_bband_df_90[f'BBU_{90}_2.0'] 
    final_df['dema30 lowest_price']  = ta.dema(final_df['lowest_price'],30)  
    final_df['dema90 lowest_price']  = ta.dema(final_df['lowest_price'],90)  
    final_df['ema90 lowest_price']  = ta.ema(final_df['lowest_price'],90)
    final_df['sma90 lowest_price']  = ta.sma(final_df['lowest_price'],90)
    final_df['tema7 lowest_price']  = ta.tema(final_df['lowest_price'],7)
    final_df['tema30 lowest_price']  = ta.tema(final_df['lowest_price'],30)
    final_df['tema90 lowest_price']  = ta.tema(final_df['lowest_price'],90) 

    final_df['dema90 avg_transaction_value']  = ta.dema(final_df['avg_transaction_value'],90)  
    final_df['ema30 avg_transaction_value']  = ta.ema(final_df['avg_transaction_value'],30)
    final_df['ema90 avg_transaction_value']  = ta.ema(final_df['avg_transaction_value'],90)
    final_df['sma90 avg_transaction_value']  = ta.sma(final_df['avg_transaction_value'],90)
    final_df['tema90 avg_transaction_value']  = ta.tema(final_df['avg_transaction_value'],90)
    final_df['wma30 avg_transaction_value']  = ta.wma(final_df['avg_transaction_value'],30)
    final_df['wma90 avg_transaction_value']  = ta.wma(final_df['avg_transaction_value'],90) 

    final_df['wma7 miner_revenue']  = ta.wma(final_df['miner_revenue'],7) 

    final_df = final_df[(final_df['Date'] >= '2013-04-01')]

    final_df = final_df[['Date', 'sma90 avg_transaction_value', 'dema90 avg_transaction_value', 
                        'closing_price', 'ema90 avg_transaction_value', 'opening_price', 'tema7 closing_price', 
                        'wma30 avg_transaction_value', 'tema90 avg_transaction_value', 'dema7 opening_price', 
                        'dema7 closing_price', 'ema30 avg_transaction_value', 'sma90 closing_price', 'wma90 avg_transaction_value', 
                        'wma7 miner_revenue', 'highest_price', 'sma90 lowest_price', 'bband_upper90 highest_price', 'sma90 highest_price', 
                        'tema30 lowest_price', 'ema90 closing_price', 'lowest_price', 'sma90 opening_price', 'tema90 closing_price', 
                        'tema30 closing_price', 'tema90 highest_price', 'tema90 lowest_price', 'bband_upper90 lowest_price', 
                        'ema90 opening_price', 'tema7 lowest_price', 'bband_upper90 closing_price', 'ema90 highest_price', 'dema90 highest_price', 
                        'dema30 lowest_price', 'tema30 highest_price', 'bband_upper30 closing_price', 'wma30 highest_price', 'bband_upper7 highest_price', 
                        'tema30 opening_price', 'bband_upper90 opening_price', 'dema90 lowest_price', 'sma30 highest_price', 'bband_lower7 opening_price', 
                        'bband_upper30 highest_price', 'dema30 opening_price', 'dema30 closing_price', 'dema30 highest_price', 'ema90 lowest_price', 'dema90 closing_price', 
                        'sma7 number_of_coins_in_circulation', 'tema7 highest_price',]]

    sgd_reg = pickle.load(open('/content/linear_reg_25.sav', 'rb'))

    X = final_df.drop(['Date'],axis=1)

    scaler = RobustScaler()
    X_scaled = X.copy()
    X_scaled[X.columns] = scaler.fit_transform(X[X.columns])

    scaler = MinMaxScaler()
    X_scaled[X.columns] =  scaler.fit_transform(X_scaled[X.columns])
    X_scaled

    today_btc_closing_price =  sgd_reg.predict(X_scaled.values[-1].reshape(1,-1))
    return today_btc_closing_price

def get_current_high_price():
    return investpy.get_crypto_historical_data(crypto='bitcoin',from_date='01/01/2021',to_date=datetime.today().strftime('%d/%m/%Y'))['Close'][-1]
