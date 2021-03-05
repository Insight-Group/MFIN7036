# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:47:57 2021

@author: Sophie
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pytz
import statsmodels.formula.api as smf

def return_calculation(df_stock):
    daily_return = [0,]
    length = len(df_stock['Adj Close'])
    
    for price in range(1,length):
        result = (df_stock['Adj Close'][price]-df_stock['Adj Close'][price-1]
                  )/df_stock['Adj Close'][price-1]
        daily_return.append(result)
        
    df_stock['Date'] = [datetime.strptime(d, '%Y-%m-%d').date() for d in df_stock['Date']]  
    df_stock['daily_return'] = pd.Series(daily_return)
    df_stock['cum_ret'] = (df_stock['daily_return'] + 1).cumprod()-1
    
    return df_stock

def polarity_calculation(df_data):   
    
    Date = []
    for each in df_data['Timestamp']:
        date = each.split('T')[0]
        date = datetime.strptime(date, '%Y-%m-%d').date()
        Date.append(date)
    
    df_data['Date'] = pd.Series(Date)
    df_polarity = (df_data.groupby(['Date']).apply(lambda x: pd.Series({'Polarity': x['TextBlob_polarity'].sum()}))
                   ).reset_index()
    
    # df_polarity['cum_pol'] = (df_polarity['Polarity'] + 1).cumprod()-1
    
    return df_polarity
    

def graphical_regression(df_test):
    
    plt.rcParams["figure.figsize"] = (12,8)
    ax = (df_test
        .assign(date=df_test['Date'], momentum=df_test['cum_ret']+1, polarity=df_test['Polarity'])
        .plot(x='Date', y=['cum_ret', 'Polarity'])
    ).set_ylabel('test result')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    # plt.savefig("./regression-result/graph_cum_ret_and_polarity.png")

def OLS_regression(df_test):
    print(smf.ols('cum_ret ~ Polarity', data=df_test).fit().summary())
    
    
    
if __name__ == '__main__':
    
    df = pd.read_csv("../dataset/test.csv", index_col=False)
    
    # convert string to timestamp (e.g. "2020-01-13T16:21:57.000Z" to Timestamp('2020-01-13 16:21:57'))
    df["Date"] = [" ".join(line.split("T")).rstrip("Z") for line in df.Timestamp]
    df["Date"] = [datetime.strptime(line, '%Y-%m-%d %H:%M:%S.%f') for line in df.Date]
    
    # change to stock trading timezone
    timezone = pytz.timezone("Europe/London")
    df["Date"] = [timezone.localize(line) for line in df.Date]
    df["Date"] = [line.astimezone(pytz.timezone('Asia/Shanghai')) for line in df.Date]
    df["Date"] = [line-timedelta(hours=9) for line in df.Date]
    
    df_daily_polarity = df.groupby([df['Date'].dt.date])['NLTK_Vader_polarity_score'].sum().reset_index()
    # df_daily_count = df.groupby([df['Date'].dt.date, 'NLTK_Vader_polarity']).count()
    
      
    # Shanghai: 600196
    df_600196 = pd.read_csv("../dataset/fusun-pharma-2020-stock-dataset/600196.csv", index_col=False)
    df_600196 = return_calculation(df_600196)
    
    df_600196_polarity = pd.merge(df_600196, df_daily_polarity, how='outer', on=['Date']).sort_values(by='Date')
    
    df_600196_polarity['daily_return_lag1'] = df_600196_polarity['daily_return'].shift(1)
    
    plt.rcParams["figure.figsize"] = (12,8)
    ax = (df_600196_polarity
        .assign(date=df_600196_polarity['Date'], momentum=df_600196_polarity['daily_return'], polarity=df_600196_polarity['NLTK_Vader_polarity_score'])
        .plot(x='Date', y=['daily_return', 'NLTK_Vader_polarity_score'])
    ).set_ylabel('test result')
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
        
    print(smf.ols('daily_return_lag1 ~ NLTK_Vader_polarity_score', data=df_600196_polarity).fit().summary())
    
    # HK: 2196
    df_2196 = pd.read_csv("../dataset/fusun-pharma-2020-stock-dataset/2196.csv", index_col=False)
    df_2196 = return_calculation(df_2196)
    