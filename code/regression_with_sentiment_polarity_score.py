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
    # df_stock['cum_ret'] = (df_stock['daily_return'] + 1).cumprod()-1
    
    return df_stock


def polarity_calculation(df_data, trading_timezone, trading_timeshift):     
    
    # convert string to timestamp (e.g. "2020-01-13T16:21:57.000Z" to Timestamp('2020-01-13 16:21:57'))
    df_data['Date'] = [" ".join(line.split("T")).rstrip("Z") for line in df_data.Timestamp]
    df_data['Date'] = [datetime.strptime(line, '%Y-%m-%d %H:%M:%S.%f') for line in df_data.Date]
    
    # change to stock trading timezone
    timezone = pytz.timezone("Europe/London") # original time zone
    df_data['Date'] = [timezone.localize(line) for line in df_data.Date]
    df_data['Date'] = [line.astimezone(pytz.timezone(trading_timezone)) for line in df_data.Date]
    df_data['Date'] = [line+timedelta(hours=int(trading_timeshift)) for line in df_data.Date]
    
    df_daily_polarity = df_data.groupby([df_data['Date'].dt.date])['NLTK_Vader_polarity_score'].sum().reset_index()
    # df_daily_count = df.groupby([df['Date'].dt.date, 'NLTK_Vader_polarity']).count()
    
    df_daily_polarity.dropna(subset = ["Date"], inplace=True)
    
    return df_daily_polarity


def lag_strategy(df_return, df_polarity, merge_on, shift_column, shift_mode, shift_period):
    
    df_test = pd.merge(df_return, df_polarity, how='outer', on=merge_on).sort_values(by=merge_on)
    df_test[shift_column + '_' + shift_mode + str(shift_period)] = df_test[shift_column].shift(shift_period)
        
    return df_test

def graphical_regression(df_test, time_series, y_return, y_score):
    
    plt.rcParams["figure.figsize"] = (12,8)
    
    df_test[y_score + '_modify'] = df_test[y_score]/(df_test[y_score].max())
    
    ax1 = (df_test
        .assign(date=df_test[time_series], y1=df_test[y_return], y2=df_test[y_score + '_modify'])
        .plot(x=time_series, y=[y_return, y_score + '_modify'])
        )
    plt.title("Stock Return vs Sentiment Polarity Score ")
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    # plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    # plt.gcf().autofmt_xdate()

    # ax1 = (df_test
    #     .assign(date=df_test['Date'], momentum=df_test['cum_ret']+1, polarity=df_test['NLTK_Vader_polarity_score'])
    #     .plot(x='Date', y=['cum_ret', 'NLTK_Vader_polarity_score'])
    #     )
    # plt.title("Cumularive Return vs Sentiment Polarity Score ")
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    # plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    # plt.gcf().autofmt_xdate()
    # # plt.savefig("./regression-result/graph_cum_ret_and_polarity.png")    
    

def OLS_regression(df_test, dep_variable, indep_variable):
    print(smf.ols('{} ~ {}'.format(dep_variable, indep_variable) , data=df_test.dropna()).fit().summary())



    
    
if __name__ == '__main__':
    
    
    # polarity
    df_twitter = pd.read_csv("../dataset/test - fusun-pharma.csv", index_col=False)   
    df_daily_polarity = polarity_calculation(df_twitter, 'Asia/Shanghai', +9)     
     
    # stock - Shanghai: 600196
    df_600196 = pd.read_csv("../dataset/fusun-pharma-2020-stock-dataset/600196.csv", index_col=False)
    df_daily_return = return_calculation(df_600196)
    
    
    # # stock - HK: 2196
    # df_2196 = pd.read_csv("../dataset/fusun-pharma-2020-stock-dataset/2196.csv", index_col=False)
    # df_2196 = return_calculation(df_2196)
   
    # # regression strategy----------------------------------------------------


    # 1. Daily lag strategy - stock return to sentiment
    df_daily = lag_strategy(df_daily_return, df_daily_polarity, merge_on='Date', 
                            shift_column='daily_return', shift_mode='lag', shift_period=1)   
        
    graphical_regression(df_daily, time_series='Date', y_return='daily_return_lag1', y_score='NLTK_Vader_polarity_score')
    OLS_regression(df_daily, dep_variable='NLTK_Vader_polarity_score', indep_variable='daily_return_lag1')
    
        
       
    # 2. Weekly strategy
    df_daily_return["year_week"] = [date.isocalendar()[:2] for date in df_600196.Date] 
    df_daily_polarity["year_week"] = [date.isocalendar()[:2] for date in df_daily_polarity.Date] 

        
    df_weekly_return = ((df_600196.groupby(["year_week"])
                            ).apply(lambda x: pd.Series({'weekly_cum_ret': ((x['daily_return'] + 1).product()-1)}))
                          ).reset_index() 
    
    df_weekly_polarity = ((df_daily_polarity.groupby(["year_week"])
                            ).apply(lambda x: pd.Series({'weekly_score': x['NLTK_Vader_polarity_score'].sum()}))
                          ).reset_index()    
    
    # 2.1 no lag
    df_weekly = pd.merge(df_weekly_return, df_weekly_polarity, how='outer', on=['year_week']).sort_values(by='year_week')  
    graphical_regression(df_weekly, time_series='year_week', y_return='weekly_cum_ret', y_score='weekly_score')
    OLS_regression(df_weekly, dep_variable='weekly_cum_ret', indep_variable='weekly_score')
    OLS_regression(df_weekly, dep_variable='weekly_score', indep_variable='weekly_cum_ret')
    
    # 2.2 return predict sentiment
    df_weekly = lag_strategy(df_weekly_return, df_weekly_polarity, merge_on='year_week',
                             shift_column='weekly_cum_ret', shift_mode='lag', shift_period=1)
    graphical_regression(df_weekly, time_series='year_week', y_return='weekly_cum_ret_lag1', y_score='weekly_score')
    OLS_regression(df_weekly, dep_variable='weekly_score', indep_variable='weekly_cum_ret_lag1')
    
    # 2.3 sentiment predict return
    df_weekly = lag_strategy(df_weekly_return, df_weekly_polarity, merge_on='year_week',
                             shift_column='weekly_score', shift_mode='lag', shift_period=1)
    graphical_regression(df_weekly, time_series='year_week', y_return='weekly_cum_ret', y_score='weekly_score_lag1')
    OLS_regression(df_weekly, dep_variable='weekly_cum_ret', indep_variable='weekly_score_lag1')
    
    

    

    

    


    