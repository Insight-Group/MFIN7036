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
    df_daily_polarity['polarity_change'] = df_daily_polarity['NLTK_Vader_polarity_score']-df_daily_polarity['NLTK_Vader_polarity_score'].shift(1)
    # df_daily_count = df.groupby([df['Date'].dt.date, 'NLTK_Vader_polarity']).count()
    
    # Specific on BiliBli, one month before LIST
    start_date_obj = datetime.strptime('2018-2-28', '%Y-%m-%d').date()
    df_daily_polarity = df_daily_polarity[df_daily_polarity.Date >= start_date_obj] 

    df_daily_polarity.dropna(subset = ["Date"], inplace=True)
    

    return df_daily_polarity


def lag_strategy(df_return, df_polarity, merge_on, shift_column, shift_mode, shift_period):
    
    df_test = pd.merge(df_return, df_polarity, how='outer', on=merge_on).sort_values(by=merge_on)
    df_test[shift_column + '_' + shift_mode + str(shift_period)] = df_test[shift_column].shift(shift_period)
        
    return df_test

def graphical_regression(df_test, time_series, y_return, y_score):
    
    plt.rcParams["figure.figsize"] = (12,8)
    
    df_test[y_score + '/max({})'.format(y_score)] = df_test[y_score]/(df_test[y_score].max())
    df_test[y_return + '/max({})'.format(y_return)] = df_test[y_return]/(df_test[y_return].max())
    
    ax1 = (df_test
        .assign(date=df_test[time_series], y1=df_test[y_return + '/max({})'.format(y_return)], y2=df_test[y_score + '/max({})'.format(y_score)])
        .plot(x=time_series, y=[y_return + '/max({})'.format(y_return), y_score + '/max({})'.format(y_score)])
        )
    plt.title("Stock Return vs Sentiment Polarity Score ")
    
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    # plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    # plt.gcf().autofmt_xdate()
        
    
    df_test[y_score+'_cum']  = df_test[y_score].cumsum()
    df_test[y_return+'_cum'] = (df_test[y_return] + 1).cumprod()
    
    df_test[y_score+'_cum' + '/max({})'.format(y_score+'_cum')] = df_test[y_score+'_cum']/(df_test[y_score+'_cum'].max())
    df_test[y_return+'_cum' + '/max({})'.format(y_return+'_cum')] = df_test[y_return+'_cum']/(df_test[y_return+'_cum'].max())
    
    ax2 = (df_test
        .assign(date=df_test[time_series], y1=df_test[y_return+'_cum' + '/max({})'.format(y_return+'_cum')], y2=df_test[y_score+'_cum' + '/max({})'.format(y_score+'_cum')])
        .plot(x=time_series, y=[y_return+'_cum' + '/max({})'.format(y_return+'_cum'), y_score+'_cum' + '/max({})'.format(y_score+'_cum')])
        )
    plt.title("Stock Cumulative Return vs Sentiment Polarity Cumulative Score ")
    

def OLS_regression(df_test, dep_variable, indep_variable):
    print(smf.ols('{} ~ {}'.format(dep_variable, indep_variable) , data=df_test.dropna()).fit().summary())

    
    
if __name__ == '__main__':
    
    
    # # polarity
    # df_twitter = pd.read_csv("../dataset/test - fusun-pharma.csv", index_col=False)   
    # df_daily_polarity = polarity_calculation(df_twitter, 'Asia/Shanghai', +8)   
    
    # polarity - bilibili
    df_twitter = pd.read_csv("../dataset/test - bilibili.csv", index_col=False)   
    df_daily_polarity = polarity_calculation(df_twitter, 'America/New_York', +8) 
     
    # # # stock - Shanghai: 600196
    # # df_600196 = pd.read_csv("../dataset/fusun-pharma-2020-stock-dataset/600196.csv", index_col=False)
    # # df_daily_return = return_calculation(df_600196)
        
    # # # stock - HK: 2196
    # # df_2196 = pd.read_csv("../dataset/fusun-pharma-2020-stock-dataset/2196.csv", index_col=False)
    # # df_daily_return = return_calculation(df_2196)
       
    # stock - NASDAQ: BILI, bilibili
    df_BILI = pd.read_csv("../dataset/bilibili-2018-2021-stock-dataset/BILI.csv", index_col=False)
    df_daily_return = return_calculation(df_BILI)
    
   
    # Regression strategies----------------------------------------------------

    # 1. Daily strategy
       
    # 1.1 no lag
    df_daily = pd.merge(df_daily_return, df_daily_polarity, how='outer', on=['Date']).sort_values(by='Date')  
    # graphical_regression(df_daily, time_series='Date', y_return='daily_return', y_score='NLTK_Vader_polarity_score')
    # OLS_regression(df_daily, dep_variable='daily_return', indep_variable='NLTK_Vader_polarity_score')
    # print('***************1.1 no lag******************')
    
    # # 1.2 return predict sentiment lag_1
    # df_daily = lag_strategy(df_daily_return, df_daily_polarity, merge_on='Date',
    #                           shift_column='daily_return', shift_mode='lag', shift_period=1)
    # graphical_regression(df_daily, time_series='Date', y_return='daily_return_lag1', y_score='NLTK_Vader_polarity_score')
    # # OLS_regression(df_daily, dep_variable='NLTK_Vader_polarity_score', indep_variable='daily_return_lag1')
    # OLS_regression(df_daily, dep_variable='daily_return_lag1', indep_variable='NLTK_Vader_polarity_score')
    # print('***************1.2 return predict sentiment lag_1******************')

    # df_daily = lag_strategy(df_daily_return, df_daily_polarity, merge_on='Date',
    #                           shift_column='daily_return', shift_mode='lag', shift_period=2)
    # graphical_regression(df_daily, time_series='Date', y_return='daily_return_lag2', y_score='NLTK_Vader_polarity_score')
    # # OLS_regression(df_daily, dep_variable='NLTK_Vader_polarity_score', indep_variable='daily_return_lag2')
    # OLS_regression(df_daily, dep_variable='daily_return_lag2', indep_variable='NLTK_Vader_polarity_score')
    # print('***************1.2 return predict sentiment lag_2******************')

    # # 1.3 sentiment predict return
    # df_daily = lag_strategy(df_daily_return, df_daily_polarity, merge_on='Date',
    #                           shift_column='NLTK_Vader_polarity_score', shift_mode='lag', shift_period=1)
    # graphical_regression(df_daily, time_series='Date', y_return='daily_return', y_score='NLTK_Vader_polarity_score_lag1')
    # OLS_regression(df_daily, dep_variable='daily_return', indep_variable='NLTK_Vader_polarity_score_lag1')
    # print('***************1.3 sentiment predict return lag_1******************')

    # df_daily = lag_strategy(df_daily_return, df_daily_polarity, merge_on='Date',
    #                           shift_column='NLTK_Vader_polarity_score', shift_mode='lag', shift_period=2)
    # graphical_regression(df_daily, time_series='Date', y_return='daily_return', y_score='NLTK_Vader_polarity_score_lag2')
    # OLS_regression(df_daily, dep_variable='daily_return', indep_variable='NLTK_Vader_polarity_score_lag2')
    # print('***************1.3 sentiment predict return lag_2******************')
    
        
       
    # # 2. Weekly strategy
    # df_daily_return["year_week"] = [date.isocalendar()[:2] for date in df_daily_return.Date] 
    # df_daily_polarity["year_week"] = [date.isocalendar()[:2] for date in df_daily_polarity.Date] 

        
    # df_weekly_return = ((df_daily_return.groupby(["year_week"])
    #                         ).apply(lambda x: pd.Series({'weekly_cum_ret': ((x['daily_return'] + 1).product()-1)}))
    #                       ).reset_index() 
    
    # df_weekly_polarity = ((df_daily_polarity.groupby(["year_week"])
    #                         ).apply(lambda x: pd.Series({'weekly_score': x['NLTK_Vader_polarity_score'].sum()}))
    #                       ).reset_index() 
    
    # # 2.1 no lag
    # df_weekly = pd.merge(df_weekly_return, df_weekly_polarity, how='outer', on=['year_week']).sort_values(by='year_week')  
    # graphical_regression(df_weekly, time_series='year_week', y_return='weekly_cum_ret', y_score='weekly_score')
    # OLS_regression(df_weekly, dep_variable='weekly_cum_ret', indep_variable='weekly_score')
    # print('***************2.1 no lag******************')
    
    # # 2.2 return predict sentiment
    # df_weekly = lag_strategy(df_weekly_return, df_weekly_polarity, merge_on='year_week',
    #                           shift_column='weekly_cum_ret', shift_mode='lag', shift_period=1)
    # graphical_regression(df_weekly, time_series='year_week', y_return='weekly_cum_ret_lag1', y_score='weekly_score')
    # # OLS_regression(df_weekly, dep_variable='weekly_score', indep_variable='weekly_cum_ret_lag1')
    # OLS_regression(df_weekly, dep_variable='weekly_cum_ret_lag1', indep_variable='weekly_score')
    # print('***************2.1 return predict sentiment lag_1******************')

    # df_weekly = lag_strategy(df_weekly_return, df_weekly_polarity, merge_on='year_week',
    #                           shift_column='weekly_cum_ret', shift_mode='lag', shift_period=2)
    # graphical_regression(df_weekly, time_series='year_week', y_return='weekly_cum_ret_lag2', y_score='weekly_score')
    # # OLS_regression(df_weekly, dep_variable='weekly_score', indep_variable='weekly_cum_ret_lag2')
    # OLS_regression(df_weekly, dep_variable='weekly_cum_ret_lag2', indep_variable='weekly_score')
    # print('***************2.1 return predict sentiment lag_2******************')

    # # 2.3 sentiment predict return
    # df_weekly = lag_strategy(df_weekly_return, df_weekly_polarity, merge_on='year_week',
    #                           shift_column='weekly_score', shift_mode='lag', shift_period=1)
    # graphical_regression(df_weekly, time_series='year_week', y_return='weekly_cum_ret', y_score='weekly_score_lag1')
    # OLS_regression(df_weekly, dep_variable='weekly_cum_ret', indep_variable='weekly_score_lag1')
    # print('***************2.3 sentiment predict return lag_1******************')

    # df_weekly = lag_strategy(df_weekly_return, df_weekly_polarity, merge_on='year_week',
    #                           shift_column='weekly_score', shift_mode='lag', shift_period=2)
    # graphical_regression(df_weekly, time_series='year_week', y_return='weekly_cum_ret', y_score='weekly_score_lag2')
    # OLS_regression(df_weekly, dep_variable='weekly_cum_ret', indep_variable='weekly_score_lag2')
    # print('***************2.3 sentiment predict return lag_2******************')



    # # 3. Monthly strategy
    # df_monthly_return = df_daily_return.groupby(pd.DatetimeIndex(df_daily_return.Date).to_period("M")) \
    #                      .apply(lambda x: pd.Series({'monthly_cum_ret': ((x['daily_return'] + 1).product()-1)})).reset_index() 

    # df_monthly_polarity = df_daily_polarity.groupby(pd.DatetimeIndex(df_daily_polarity.Date).to_period("M")) \
    #                       .apply(lambda x: pd.Series({'monthly_score': x['NLTK_Vader_polarity_score'].sum()})).reset_index()  

    # # 3.1 no lag
    # df_monthly = pd.merge(df_monthly_return, df_monthly_polarity, how='outer', on=['Date']).sort_values(by='Date')  
    # graphical_regression(df_monthly, time_series='Date', y_return='monthly_cum_ret', y_score='monthly_score')
    # OLS_regression(df_monthly, dep_variable='monthly_cum_ret', indep_variable='monthly_score')
    # print('***************3.1 no lag******************')

    # # 3.2 return predict sentiment
    # df_monthly = lag_strategy(df_monthly_return, df_monthly_polarity, merge_on='Date',
    #                          shift_column='monthly_cum_ret', shift_mode='lag', shift_period=1)
    # graphical_regression(df_monthly, time_series='Date', y_return='monthly_cum_ret_lag1', y_score='monthly_score')
    # # OLS_regression(df_monthly, dep_variable='monthly_score', indep_variable='monthly_cum_ret_lag1')
    # OLS_regression(df_monthly, dep_variable='monthly_cum_ret_lag1', indep_variable='monthly_score')
    # print('***************3.2 return predict sentiment lag_1******************')

    # df_monthly = lag_strategy(df_monthly_return, df_monthly_polarity, merge_on='Date',
    #                          shift_column='monthly_cum_ret', shift_mode='lag', shift_period=2)
    # graphical_regression(df_monthly, time_series='Date', y_return='monthly_cum_ret_lag2', y_score='monthly_score')
    # # OLS_regression(df_monthly, dep_variable='monthly_score', indep_variable='monthly_cum_ret_lag2')
    # OLS_regression(df_monthly, dep_variable='monthly_cum_ret_lag2', indep_variable='monthly_score')
    # print('***************3.2 return predict sentiment lag_2******************')

    # # 3.3 sentiment predict return
    # df_monthly = lag_strategy(df_monthly_return, df_monthly_polarity, merge_on='Date',
    #                          shift_column='monthly_score', shift_mode='lag', shift_period=1)
    # graphical_regression(df_monthly, time_series='Date', y_return='monthly_cum_ret', y_score='monthly_score_lag1')
    # OLS_regression(df_monthly, dep_variable='monthly_cum_ret', indep_variable='monthly_score_lag1')
    # print('***************3.3 sentiment predict return lag_1******************')

    # df_monthly = lag_strategy(df_monthly_return, df_monthly_polarity, merge_on='Date',
    #                          shift_column='monthly_score', shift_mode='lag', shift_period=2)
    # graphical_regression(df_monthly, time_series='Date', y_return='monthly_cum_ret', y_score='monthly_score_lag2')
    # OLS_regression(df_monthly, dep_variable='monthly_cum_ret', indep_variable='monthly_score_lag2')
    # print('***************3.3 sentiment predict return lag_2******************')
    
    
    # Further study------------------------------------------------------------------------------
    
    # df_daily_polarity['day'] = [date.isocalendar()[1] for date in df_daily_return.Date]
    
    polarity_buy=20
    polarity_sell =-1
    return_buy=0.02
    return_sell=0.1
    
    # # df_daily['point']=''
    # # df_daily['point'] = ['buy' for in df_daily if df_daily[(df_daily['polarity_change'] > x) & (df_daily['daily_return'] < y)]]
    # # df_daily.loc[(df_daily['polarity_change'] > 1) & (df_daily['daily_return'] < 0.1)]['point'] = pd.Series(['buy'])
    # # df_daily['point'] = [i for i in df_daily['polarity_change'] j in df_daily['daily_return'] if (i > x) & (j < y)]
    
    Trade=[]
    df_daily = df_daily.reset_index()
    for i in range(len(df_daily)-1):
        if ((df_daily['polarity_change'].values[i] > polarity_buy) & (df_daily['daily_return'].values[i] < return_buy)):
            # print("Trade Call for {row} is Buy.".format(row=data_amd.index[i].date()))
            Trade.append('buy')
        elif ((df_daily['polarity_change'].values[i] < polarity_sell) & (df_daily['daily_return'].values[i] > return_sell)):
            # print("Trade Call for {row} is Sell.".format(row=data_amd.index[i].date()))
            Trade.append('sell')
        else:
            Trade.append('')
            
    df_daily['Trade'] = pd.Series(Trade)
    
    
    df_daily.to_csv("./test.csv")