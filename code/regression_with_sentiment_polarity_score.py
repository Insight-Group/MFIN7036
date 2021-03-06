# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:47:57 2021

@author: Sophie
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
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


def polarity_calculation(df_data, trading_timezone, trading_timeshift):     
    
    # convert string to timestamp (e.g. "2020-01-13T16:21:57.000Z" to Timestamp('2020-01-13 16:21:57'))
    df_data["Date"] = [" ".join(line.split("T")).rstrip("Z") for line in df_data.Timestamp]
    df_data["Date"] = [datetime.strptime(line, '%Y-%m-%d %H:%M:%S.%f') for line in df_data.Date]
    
    # change to stock trading timezone
    timezone = pytz.timezone("Europe/London") # original time zone
    df["Date"] = [timezone.localize(line) for line in df.Date]
    df["Date"] = [line.astimezone(pytz.timezone(trading_timezone)) for line in df.Date]
    df["Date"] = [line+timedelta(hours=int(trading_timeshift)) for line in df.Date]
    
    df_daily_polarity = df.groupby([df['Date'].dt.date])['NLTK_Vader_polarity_score'].sum().reset_index()
    # df_daily_count = df.groupby([df['Date'].dt.date, 'NLTK_Vader_polarity']).count()
    
    return df_daily_polarity
    

def graphical_regression(df_test):
    
    plt.rcParams["figure.figsize"] = (12,8)
    ax1 = (df_test
        .assign(date=df_test['Date'], momentum=df_test['cum_ret']+1, polarity=df_test['NLTK_Vader_polarity_score'])
        .plot(x='Date', y=['cum_ret', 'NLTK_Vader_polarity_score'])
        )
    plt.title("Cumularive Return vs Sentiment Polarity Score ")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    # plt.savefig("./regression-result/graph_cum_ret_and_polarity.png")
    
    
    ax2 = (df_test
        .assign(date=df_test['Date'], momentum=df_test['daily_return'], polarity=df_test['NLTK_Vader_polarity_score'])
        .plot(x='Date', y=['daily_return', 'NLTK_Vader_polarity_score'])
        )
    plt.title("Daily Return vs Sentiment Polarity Score ")
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.gcf().autofmt_xdate()
    
    

def OLS_regression(df_test, y, x):
    print(smf.ols('{} ~ {}'.format(y,x) , data=df_test.dropna()).fit().summary())
    
    
    
if __name__ == '__main__':
    
    
    # polarity
    df = pd.read_csv("../dataset/test - fusun-pharma.csv", index_col=False)
    df_daily_polarity = polarity_calculation(df, 'Asia/Shanghai', +9)
     
    # stock - Shanghai: 600196
    df_600196 = pd.read_csv("../dataset/fusun-pharma-2020-stock-dataset/600196.csv", index_col=False)
    df_600196 = return_calculation(df_600196)
    # stock - HK: 2196
    df_2196 = pd.read_csv("../dataset/fusun-pharma-2020-stock-dataset/2196.csv", index_col=False)
    df_2196 = return_calculation(df_2196)
   
    # regression---------------------------------------------------------------
    df_600196_polarity = pd.merge(df_600196, df_daily_polarity, how='outer', on=['Date']).sort_values(by='Date')
    
    # graphical overview
    graphical_regression(df_600196_polarity)
    
    # ols: stock return to sentiment
    df_600196_polarity['daily_return_lag1'] = df_600196_polarity['daily_return'].shift(1)
    OLS_regression(df_600196_polarity, 'NLTK_Vader_polarity_score', 'daily_return_lag1')
        
    
    # df_data['Date'] = pd.Series(Date)
    # df_polarity = (df_data.groupby(['Date']).apply(lambda x: pd.Series({'Polarity': x['TextBlob_polarity'].sum()}))
    #                ).reset_index()
    
    # df_polarity['cum_pol'] = (df_polarity['Polarity'] + 1).cumprod()-1
    