# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 20:00:43 2021

@author: 78746
"""

import data_reshaping_and_preprocessing as drp
import pandas as pd
import numpy as np
import datetime
# Dataframe with columns - UserScreenName,UserName,Timestamp,Text,Emojis,Comments,Likes,Retweets,Image link,Tweet URL
data_frame = drp.read_file("../dataset/fusun-pharma-2020-twitter-dataset/fosun_pharma_2020.csv")

# Column - word_tokens and sent_tokens added to dataframe
data_frame = drp.tokenize_and_add_column(data_frame)
# Column - tfdif added to dataframe 
data_frame = drp.calculate_tfidf_and_add_column(data_frame)

from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import preprocessing
from sklearn import utils  
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

def regression_SVM_daily_tweets(df_data):
    Date = []
    for each in df_data['Timestamp']:
        date = each.split('T')[0]
        date = datetime.strptime(date, '%Y-%m-%d').date()
        Date.append(date)
    
    df_data['Date'] = pd.Series(Date)
    df_data['word_tokens']=[" ".join(word_tokens) for word_tokens in df_data['word_tokens']]
    df_polarity = (df_data.groupby(['Date']).apply(lambda x: pd.Series({'daily_tweets':" ".join(x['word_tokens'])}))
                    ).reset_index()
    
    return df_polarity

regression_SVM = regression_SVM_daily_tweets(data_frame)

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import statsmodels.formula.api as smf



Fosun_stock_price = pd.read_csv("../dataset/fusun-pharma-2020-stock-price/fosun_pharma_2020.csv")

list1=[]

for i in range(1,31):
    range_test = pd.concat([Fosun_stock_price.iloc[i:]['Adj Close'],pd.Series([0]*i)])
    range_test = range_test.reset_index(drop=True)
    df_range = 'range_test_'+str(i)+'_days'
    Fosun_stock_price[df_range]=pd.Series(range_test)
    df_volatility = 'Volatility_'+str(i)+'_days'
    regression_SVM[df_range] = Fosun_stock_price[df_range]
    regression_SVM['df_volatility']=np.where(regression_SVM[df_range]/regression_SVM[df_range].shift(1)-1>0,1,0)
    y=regression_SVM['df_volatility']
    
    v=TfidfVectorizer(stop_words='english',max_df=0.9)
    X = v.fit_transform(regression_SVM['daily_tweets'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 40)
    classifier_linear=svm.SVC(kernel='linear')
    classifier_linear.fit(X_train,y_train)
    
    prediction_linear = classifier_linear.predict(X_test)
    report = classification_report(y_test,prediction_linear,output_dict = True)
    list1.append(report['weighted avg']['precision'])
    
    print(list1)

plt.show()
plt.bar(list(range(1,31)),list1)
plt.title('SVM model prediction precision')
plt.xlabel('# of days')
plt.ylabel('Precision')





