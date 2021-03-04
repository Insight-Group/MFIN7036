# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:52:41 2021

@author: Sophie
"""

import os
import pandas as pd
import nltk
import NLTK_Vader

# from nltk.sentiment.vader import SentimentIntensityAnalyzer   # NLTK Vader
# from textblob import TextBlob                                 # TextBlob
# C:\Users\SAMSUNG-1\AppData\Roaming\nltk_data\


"""analyse sentiment by NLTK Vader"""
def NLTK_Vader_sentiment_analysis(twitter_data_frame):

    sia = NLTK_Vader.SentimentIntensityAnalyzer()
        
    NLTK_Vader_polarity = []
    
    for line in twitter_data_frame['Text']:
        
        tfidf_dict = twitter_data_frame[twitter_data_frame['Text']==line]['tfidf'].tolist()[0]
        
        tweet_polarity = 0
        
        for sent in twitter_data_frame[twitter_data_frame['Text']==line]['sent_tokens'].tolist()[0]:
            sentiment = sia.polarity_scores(sent, tfidf_dict)
            tweet_polarity += sentiment['compound']
        NLTK_Vader_polarity.append(tweet_polarity)
       
    
    twitter_data_frame['NLTK_Vader_polarity'] = pd.Series(NLTK_Vader_polarity)
    
    return twitter_data_frame


# """analyse sentiment by TextBlob"""
# def TextBlob_sentiment_analysis(twitter_data_frame):
#     TextBlob_polarity = []
    
#     for line in twitter_data_frame['final_token']:
#         [polarity, subjectivity] = list(TextBlob(str(line)).sentiment)
#         TextBlob_polarity.append(polarity)
        
#     twitter_data_frame['TextBlob_polarity'] = pd.Series(TextBlob_polarity)
    
#     return twitter_data_frame