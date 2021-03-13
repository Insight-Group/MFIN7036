# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 09:47:57 2021

@author: Sophie
"""

import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
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


def polarity_calculation(df_twitter, polarity_column, trading_timezone, trading_timeshift):     
    
    # convert string to timestamp (e.g. "2020-01-13T16:21:57.000Z" to Timestamp('2020-01-13 16:21:57'))
    df_twitter['Date'] = [" ".join(date.split("T")).rstrip("Z") for date in df_twitter.Timestamp]
    df_twitter['Date'] = [datetime.strptime(date, '%Y-%m-%d %H:%M:%S.%f') for date in df_twitter.Date]
    
    # change to stock trading timezone
    timezone = pytz.timezone("Europe/London") # original time zone
    df_twitter['Date'] = [timezone.localize(date) for date in df_twitter.Date]
    df_twitter['Date'] = [date.astimezone(pytz.timezone(trading_timezone)) for date in df_twitter.Date]
    df_twitter['Date'] = [date+timedelta(hours=int(trading_timeshift)) for date in df_twitter.Date]
    
    # calculate the sum of sentiment polarity scores of tweets released on each day
    df_daily_polarity_score = df_twitter.groupby([df_twitter['Date'].dt.date])[polarity_column].sum().reset_index()
    df_daily_polarity_score['polarity_score_change'] = df_daily_polarity_score[polarity_column]-df_daily_polarity_score[polarity_column].shift(1)
    
    # count volumes of tweets with positive, negative and neutral sentiment on each day
    df_daily_polarity_count = df_twitter.groupby([df_twitter['Date'].dt.date]).apply(lambda x: pd.Series({'pos_count': (x[polarity_column]>0).sum(),
                                                                                                          'neg_count': (x[polarity_column]<0).sum(),
                                                                                                          'neu_count': (x[polarity_column]==0).sum(),
                                                                                                          'tweet_count': x[polarity_column].count()
                                                                                                          })).reset_index()    
    # bringing polarity results together
    df_daily_polarity = pd.merge(df_daily_polarity_score, df_daily_polarity_count, on='Date')
                                                  
    
    # Specific on BiliBli, one month before LIST
    start_date_obj = datetime.strptime('2018-2-28', '%Y-%m-%d').date()
    df_daily_polarity = df_daily_polarity[df_daily_polarity.Date >= start_date_obj] 

    df_daily_polarity.dropna(subset = ['Date'], inplace=True)

    return df_daily_polarity


def lag_strategy(df_return, df_polarity, merge_on, shift_column, shift_mode, shift_period):
    
    df_test = pd.merge(df_return, df_polarity, how='outer', on=merge_on).sort_values(by=merge_on)
    df_test[shift_column + '_' + shift_mode + str(shift_period)] = df_test[shift_column].shift(shift_period)
        
    return df_test


def graphical_regression(df_test, time_series, y_return, y_score):
    
    plt.rcParams["figure.figsize"] = (12,8)
    
    # normalization
    df_test[y_return + '/max({})'.format(y_return)] = df_test[y_return]/(df_test[y_return].max())
    df_test[y_score + '/max({})'.format(y_score)] = df_test[y_score]/(df_test[y_score].max())
    df_test['pos_count/max(tweet_count)'] = df_test['pos_count']/(df_test['tweet_count'].max())
    df_test['neg_count/max(tweet_count)'] = df_test['neg_count']/(df_test['tweet_count'].max())
    df_test['tweet_count/max(tweet_count)'] = df_test['tweet_count']/(df_test['tweet_count'].max())
    
    ax1 = (df_test
        .assign(date=df_test[time_series], y1=df_test[y_return + '/max({})'.format(y_return)], y2=df_test[y_score + '/max({})'.format(y_score)],
                y3=df_test['pos_count/max(tweet_count)'], y4=df_test['neg_count/max(tweet_count)'], y5=df_test['tweet_count/max(tweet_count)'])
        .plot(x=time_series, y=[y_return + '/max({})'.format(y_return), y_score + '/max({})'.format(y_score), 
                                'pos_count/max(tweet_count)', 'neg_count/max(tweet_count)', 'tweet_count/max(tweet_count)'])
        )
    plt.title("Stock Return vs Sentiment Polarity Score and Counts")
    
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
    # plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    # plt.gcf().autofmt_xdate()
    
    # calculate the cumulative value-----------------------------------------------------------------
    df_test[y_return+'_cum'] = (df_test[y_return] + 1).cumprod()
    df_test[y_score+'_cum']  = df_test[y_score].cumsum()
    df_test['pos_count_cum'] = df_test['pos_count'].cumsum()
    df_test['neg_count_cum'] = df_test['neg_count'].cumsum()
    df_test['tweet_count_cum'] = df_test['tweet_count'].cumsum()
    # normalization
    df_test[y_score+'_cum' + '/max({})'.format(y_score+'_cum')] = df_test[y_score+'_cum']/(df_test[y_score+'_cum'].max())
    df_test[y_return+'_cum' + '/max({})'.format(y_return+'_cum')] = df_test[y_return+'_cum']/(df_test[y_return+'_cum'].max())
    df_test['pos_count_cum/max(tweet_count_cum)'] = df_test['pos_count_cum']/(df_test['tweet_count_cum'].max())
    df_test['neg_count_cum/max(tweet_count_cum)'] = df_test['neg_count_cum']/(df_test['tweet_count_cum'].max())
    df_test['tweet_count_cum/max(tweet_count_cum)'] = df_test['tweet_count_cum']/(df_test['tweet_count_cum'].max())
    
    ax2 = (df_test
        .assign(date=df_test[time_series], y1=df_test[y_return+'_cum' + '/max({})'.format(y_return+'_cum')], y2=df_test[y_score+'_cum' + '/max({})'.format(y_score+'_cum')],
                y3=df_test['pos_count_cum/max(tweet_count_cum)'], y4=df_test['neg_count_cum/max(tweet_count_cum)'], y5=df_test['tweet_count_cum/max(tweet_count_cum)'])
        .plot(x=time_series, y=[y_return+'_cum' + '/max({})'.format(y_return+'_cum'), y_score+'_cum' + '/max({})'.format(y_score+'_cum'),
                                'pos_count_cum/max(tweet_count_cum)', 'neg_count_cum/max(tweet_count_cum)', 'tweet_count_cum/max(tweet_count_cum)'])
        )
    plt.title("Stock Cumulative Return vs Sentiment Polarity Cumulative Score and Counts")
    

def OLS_regression(df_test, return_column, polarity_score_column, dependent_variable):
    
    if dependent_variable == return_column:
        # polarity results are independent variables
        print(smf.ols('{} ~ {} + pos_count + neg_count + tweet_count'.format(return_column, polarity_score_column), data=df_test.dropna()).fit().summary())
        
    elif dependent_variable == polarity_score_column:
        # print(smf.ols('{} ~ {}'.format(polarity_score_column, return_column) , data=df_test.dropna()).fit().summary())
        # switch the positions of dependent variable and independent variable to make the output beta value comparable in all the strategies
        print(smf.ols('{} ~ {}'.format(return_column, polarity_score_column) , data=df_test.dropna()).fit().summary())
        
    else:
        print('Incorrect dependent_variable')
    
    
    
# The main regression strategies-----------------------------------------------


def regression_strategy(df_stock, df_twitter, polarity_column, trading_timezone, trading_timeshift):
    
    df_daily_polarity = polarity_calculation(df_twitter, polarity_column, trading_timezone, trading_timeshift)
    df_daily_return = return_calculation(df_stock)

    # 1. Daily strategy
       
    # 1.1 no lag
    df_daily = pd.merge(df_daily_return, df_daily_polarity, how='outer', on=['Date']).sort_values(by='Date')  
    graphical_regression(df_daily, time_series='Date', y_return='daily_return', y_score=polarity_column)
    OLS_regression(df_daily, 'daily_return', polarity_column, dependent_variable='daily_return')
    print('***************1.1 no lag******************')
    
    # 1.2 return predict sentiment lag_1
    df_daily = lag_strategy(df_daily_return, df_daily_polarity, merge_on='Date',
                              shift_column='daily_return', shift_mode='lag', shift_period=1)
    graphical_regression(df_daily, time_series='Date', y_return='daily_return_lag1', y_score=polarity_column)
    OLS_regression(df_daily, 'daily_return_lag1', polarity_column, dependent_variable=polarity_column)
    print('***************1.2 return predict sentiment lag_1******************')

    df_daily = lag_strategy(df_daily_return, df_daily_polarity, merge_on='Date',
                              shift_column='daily_return', shift_mode='lag', shift_period=2)
    graphical_regression(df_daily, time_series='Date', y_return='daily_return_lag2', y_score=polarity_column)
    OLS_regression(df_daily, 'daily_return_lag2', polarity_column, dependent_variable=polarity_column)
    print('***************1.2 return predict sentiment lag_2******************')

    # 1.3 sentiment predict return
    df_daily = lag_strategy(df_daily_return, df_daily_polarity, merge_on='Date',
                              shift_column=polarity_column, shift_mode='lag', shift_period=1)
    graphical_regression(df_daily, time_series='Date', y_return='daily_return', y_score=polarity_column+'_lag1')
    OLS_regression(df_daily, 'daily_return', polarity_column+'_lag1', dependent_variable='daily_return')
    print('***************1.3 sentiment predict return lag_1******************')

    df_daily = lag_strategy(df_daily_return, df_daily_polarity, merge_on='Date',
                              shift_column=polarity_column, shift_mode='lag', shift_period=2)
    graphical_regression(df_daily, time_series='Date', y_return='daily_return', y_score=polarity_column+'_lag2')
    OLS_regression(df_daily, 'daily_return', polarity_column+'_lag2', dependent_variable='daily_return')
    print('***************1.3 sentiment predict return lag_2******************')
    
        
       
    # 2. Weekly strategy
    df_daily_return["year_week"] = [date.isocalendar()[:2] for date in df_daily_return.Date] 
    df_daily_polarity["year_week"] = [date.isocalendar()[:2] for date in df_daily_polarity.Date] 

        
    df_weekly_return = df_daily_return.groupby(["year_week"])\
                        .apply(lambda x: pd.Series({'weekly_ret': ((x['daily_return'] + 1).product()-1)})).reset_index() 
    
    df_weekly_polarity = df_daily_polarity.groupby(["year_week"])\
                          .apply(lambda x: pd.Series({'weekly_score': x[polarity_column].sum(),
                                                      'pos_count': (x[polarity_column]>0).sum(),
                                                      'neg_count': (x[polarity_column]<0).sum(),
                                                      # 'neu_count': (x[polarity_column]==0).sum(),
                                                      'tweet_count': x[polarity_column].count()})).reset_index() 
    
    # 2.1 no lag
    df_weekly = pd.merge(df_weekly_return, df_weekly_polarity, how='outer', on=['year_week']).sort_values(by='year_week')  
    graphical_regression(df_weekly, time_series='year_week', y_return='weekly_ret', y_score='weekly_score')
    OLS_regression(df_weekly, 'weekly_ret', 'weekly_score', dependent_variable='weekly_ret')
    print('***************2.1 no lag******************')
    
    # 2.2 return predict sentiment
    df_weekly = lag_strategy(df_weekly_return, df_weekly_polarity, merge_on='year_week',
                              shift_column='weekly_ret', shift_mode='lag', shift_period=1)
    graphical_regression(df_weekly, time_series='year_week', y_return='weekly_ret_lag1', y_score='weekly_score')
    OLS_regression(df_weekly, 'weekly_ret_lag1', 'weekly_score', dependent_variable='weekly_score')
    print('***************2.1 return predict sentiment lag_1******************')

    df_weekly = lag_strategy(df_weekly_return, df_weekly_polarity, merge_on='year_week',
                              shift_column='weekly_ret', shift_mode='lag', shift_period=2)
    graphical_regression(df_weekly, time_series='year_week', y_return='weekly_ret_lag2', y_score='weekly_score')
    OLS_regression(df_weekly, 'weekly_ret_lag2', 'weekly_score', dependent_variable='weekly_score')
    print('***************2.1 return predict sentiment lag_2******************')

    # 2.3 sentiment predict return
    df_weekly = lag_strategy(df_weekly_return, df_weekly_polarity, merge_on='year_week',
                              shift_column='weekly_score', shift_mode='lag', shift_period=1)
    graphical_regression(df_weekly, time_series='year_week', y_return='weekly_ret', y_score='weekly_score_lag1')
    OLS_regression(df_weekly, 'weekly_ret', 'weekly_score_lag1', dependent_variable='weekly_ret')
    print('***************2.3 sentiment predict return lag_1******************')

    df_weekly = lag_strategy(df_weekly_return, df_weekly_polarity, merge_on='year_week',
                              shift_column='weekly_score', shift_mode='lag', shift_period=2)
    graphical_regression(df_weekly, time_series='year_week', y_return='weekly_ret', y_score='weekly_score_lag2')
    OLS_regression(df_weekly, 'weekly_ret', 'weekly_score_lag2', dependent_variable='weekly_ret')
    print('***************2.3 sentiment predict return lag_2******************')



    # 3. Monthly strategy
    df_monthly_return = df_daily_return.groupby(pd.DatetimeIndex(df_daily_return.Date).to_period("M")) \
                          .apply(lambda x: pd.Series({'monthly_ret': ((x['daily_return'] + 1).product()-1)})).reset_index() 

    df_monthly_polarity = df_daily_polarity.groupby(pd.DatetimeIndex(df_daily_polarity.Date).to_period("M")) \
                          .apply(lambda x: pd.Series({'monthly_score': x[polarity_column].sum(),
                                                      'pos_count': (x[polarity_column]>0).sum(),
                                                      'neg_count': (x[polarity_column]<0).sum(),
                                                      # 'neu_count': (x[polarity_column]==0).sum(),
                                                      'tweet_count': x[polarity_column].count()})).reset_index()  

    # 3.1 no lag
    df_monthly = pd.merge(df_monthly_return, df_monthly_polarity, how='outer', on=['Date']).sort_values(by='Date')  
    graphical_regression(df_monthly, time_series='Date', y_return='monthly_ret', y_score='monthly_score')
    OLS_regression(df_monthly, 'monthly_ret', 'monthly_score', dependent_variable='monthly_ret')
    print('***************3.1 no lag******************')

    # 3.2 return predict sentiment
    df_monthly = lag_strategy(df_monthly_return, df_monthly_polarity, merge_on='Date',
                              shift_column='monthly_ret', shift_mode='lag', shift_period=1)
    graphical_regression(df_monthly, time_series='Date', y_return='monthly_ret_lag1', y_score='monthly_score')
    OLS_regression(df_monthly, 'monthly_ret_lag1', 'monthly_score', dependent_variable='monthly_score')
    print('***************3.2 return predict sentiment lag_1******************')

    df_monthly = lag_strategy(df_monthly_return, df_monthly_polarity, merge_on='Date',
                              shift_column='monthly_ret', shift_mode='lag', shift_period=2)
    graphical_regression(df_monthly, time_series='Date', y_return='monthly_ret_lag2', y_score='monthly_score')
    OLS_regression(df_monthly, 'monthly_ret_lag2', 'monthly_score', dependent_variable='monthly_score')
    print('***************3.2 return predict sentiment lag_2******************')

    # 3.3 sentiment predict return
    df_monthly = lag_strategy(df_monthly_return, df_monthly_polarity, merge_on='Date',
                              shift_column='monthly_score', shift_mode='lag', shift_period=1)
    graphical_regression(df_monthly, time_series='Date', y_return='monthly_ret', y_score='monthly_score_lag1')
    OLS_regression(df_monthly, 'monthly_ret', 'monthly_score_lag1', dependent_variable='monthly_ret')
    print('***************3.3 sentiment predict return lag_1******************')

    df_monthly = lag_strategy(df_monthly_return, df_monthly_polarity, merge_on='Date',
                              shift_column='monthly_score', shift_mode='lag', shift_period=2)
    graphical_regression(df_monthly, time_series='Date', y_return='monthly_ret', y_score='monthly_score_lag2')
    OLS_regression(df_monthly, 'monthly_ret', 'monthly_score_lag2', dependent_variable='monthly_ret')
    print('***************3.3 sentiment predict return lag_2******************')
    
    
    # # Further study------------------------------------------------------------------------------
    
    # # df_daily_polarity['day'] = [date.isocalendar()[1] for date in df_daily_return.Date]
    
    # polarity_buy=20
    # polarity_sell =-1
    # return_buy=0.02
    # return_sell=0.1
    
    # # # df_daily['point']=''
    # # # df_daily['point'] = ['buy' for in df_daily if df_daily[(df_daily['polarity_change'] > x) & (df_daily['daily_return'] < y)]]
    # # # df_daily.loc[(df_daily['polarity_change'] > 1) & (df_daily['daily_return'] < 0.1)]['point'] = pd.Series(['buy'])
    # # # df_daily['point'] = [i for i in df_daily['polarity_change'] j in df_daily['daily_return'] if (i > x) & (j < y)]
    
    # Trade=[]
    # df_daily = df_daily.reset_index()
    # for i in range(len(df_daily)-1):
    #     if ((df_daily['polarity_change'].values[i] > polarity_buy) & (df_daily['daily_return'].values[i] < return_buy)):
    #         # print("Trade Call for {row} is Buy.".format(row=data_amd.index[i].date()))
    #         Trade.append('buy')
    #     elif ((df_daily['polarity_change'].values[i] < polarity_sell) & (df_daily['daily_return'].values[i] > return_sell)):
    #         # print("Trade Call for {row} is Sell.".format(row=data_amd.index[i].date()))
    #         Trade.append('sell')
    #     else:
    #         Trade.append('')
            
    # df_daily['Trade'] = pd.Series(Trade)
    
    
    # df_daily.to_csv("./test.csv")
    
    
    
    
    
    
if __name__ == '__main__':
    
    # select polarity results from different sentiment analysis approaches
    choice = input('Use polarity results from which approach? \
                   [Options: lexicon-based approach, machine learning approach]')
                   
    best_sentiment_analysis_results = {'lexicon-based approach':    'Vader_Text_polarity',
                                       'machine learning approach': 'prediction_nb'} 
                      
    polarity_column = best_sentiment_analysis_results[choice]
    
    
    # # polarity - fusun pharma
    # df_twitter = pd.read_csv("../dataset/test - fusun-pharma.csv", index_col=False)   
    # df_daily_polarity = polarity_calculation(df_twitter, polarity_column, 'Asia/Shanghai', +8)   
    
    # polarity - bilibili
    df_twitter = pd.read_csv("../dataset/test - bilibili.csv", index_col=False)   
    df_daily_polarity = polarity_calculation(df_twitter, polarity_column, 'America/New_York', +8) 
     
    # # # stock - Shanghai: 600196
    # # df_600196 = pd.read_csv("../dataset/fusun-pharma-2020-stock-dataset/600196.csv", index_col=False)
    # # df_daily_return = return_calculation(df_600196)
        
    # # # stock - HK: 2196
    # # df_2196 = pd.read_csv("../dataset/fusun-pharma-2020-stock-dataset/2196.csv", index_col=False)
    # # df_daily_return = return_calculation(df_2196)
       
    # stock - NASDAQ: BILI, bilibili
    df_BILI = pd.read_csv("../dataset/bilibili-2018-2021-stock-dataset/BILI.csv", index_col=False)
    df_daily_return = return_calculation(df_BILI)
    
   
    