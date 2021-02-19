# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 21:52:41 2021

@author: Sophie
"""

import os
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer   # NLTK Vader
from textblob import TextBlob                           # TextBlob
# C:\Users\SAMSUNG-1\AppData\Roaming\nltk_data\

twitter_data_frame = pd.read_csv("./data_reshaping_and_preprocessing.csv", index_col=0)    


"""analyse sentiment by NLTK Vader"""

sia = SentimentIntensityAnalyzer()
    
NLTK_Vader_polarity = []

for line in twitter_data_frame['final_token']:
    sentiment = sia.polarity_scores(line)
    NLTK_Vader_polarity.append(sentiment)

twitter_data_frame['NLTK_Vader_polarity'] = pd.Series(NLTK_Vader_polarity)


"""analyse sentiment by TextBlob"""

TextBlob_polarity = []

for line in twitter_data_frame['final_token']:
    [polarity, subjectivity] = list(TextBlob(line).sentiment)
    TextBlob_polarity.append(polarity)
    
twitter_data_frame['TextBlob_polarity'] = pd.Series(TextBlob_polarity)


twitter_data_frame.to_csv("./sentiment_analysis.csv", index = False)