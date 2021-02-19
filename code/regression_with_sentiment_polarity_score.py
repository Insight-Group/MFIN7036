# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:47:57 2021

@author: Sophie
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import statsmodels.formula.api as smf

def return_calculation(df_stock):
    daily_return = [0,]
    length = len(df_stock['Adj Close'])
    
    for line in range(1,length):
        result = (df_stock['Adj Close'][line]-df_stock['Adj Close'][line-1]
                  )/df_stock['Adj Close'][line-1]
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
    

df_stock = pd.read_csv("./600196.csv")
df_stock = return_calculation(df_stock)

df_data = pd.read_csv("./sentiment_analysis.csv")
df_polarity = polarity_calculation(df_data)

df_test = pd.merge(df_stock, df_polarity, how='left', on=['Date'] )


plt.rcParams["figure.figsize"] = (12,8)
ax = (df_test
    .assign(date=df_test['Date'], momentum=df_test['cum_ret']+1, polarity=df_test['Polarity'])
    .plot(x='Date', y=['cum_ret', 'Polarity'])
).set_ylabel('test result')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()

print(smf.ols('cum_ret ~ Polarity', data=df_test).fit().summary())