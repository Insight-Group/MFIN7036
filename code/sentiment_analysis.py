# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:52:41 2021

@author: Sophie
"""

import os
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer   # NLTK Vader
from textblob import TextBlob                           # TextBlob
# C:\Users\SAMSUNG-1\AppData\Roaming\nltk_data\


"""analyse sentiment by NLTK Vader"""
def NLTK_Vader_sentiment_analysis(twitter_data_frame):

    sia = SentimentIntensityAnalyzer()
        
    NLTK_Vader_polarity = []
    
    for line in twitter_data_frame['final_token']:
        sentiment = sia.polarity_scores(str(line))
        NLTK_Vader_polarity.append(sentiment)
    
    twitter_data_frame['NLTK_Vader_polarity'] = pd.Series(NLTK_Vader_polarity)
    
    return twitter_data_frame


"""analyse sentiment by TextBlob"""
def TextBlob_sentiment_analysis(twitter_data_frame):
    TextBlob_polarity = []
    
    for line in twitter_data_frame['final_token']:
        [polarity, subjectivity] = list(TextBlob(str(line)).sentiment)
        TextBlob_polarity.append(polarity)
        
    twitter_data_frame['TextBlob_polarity'] = pd.Series(TextBlob_polarity)
    
    return twitter_data_frame