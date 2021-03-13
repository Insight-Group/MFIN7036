# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 17:16:56 2021

@author: Sophie
"""

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer   # NLTK Vader
import NLTK_Vader                                             # modified Vader
from textblob import TextBlob                                 # TextBlob  

# C:\Users\SAMSUNG-1\AppData\Roaming\nltk_data\


"""analyse sentiment by NLTK Vader"""
def NLTK_Vader_sentiment_analysis(df):

    sia = SentimentIntensityAnalyzer()

    # analyse the raw text of each tweet
    Vader_Text_polarity = []

    for tweet in df['Text']:
        sentiment = sia.polarity_scores(tweet)
        Vader_Text_polarity.append(sentiment['compound'])
    
    df['Vader_Text_polarity'] = pd.Series(Vader_Text_polarity)


    # analyse the processed text of each tweet
    Vader_processed_text_polarity = []

    for tweet in df['processed_text']:
        sentiment = sia.polarity_scores(tweet)
        Vader_processed_text_polarity.append(sentiment['compound'])
    
    df['Vader_processed_text_polarity'] = pd.Series(Vader_processed_text_polarity)
    
        
    # analyse the tokenized sentences in each tweet
    Vader_Sent_polarity = []
    
    for tweet in df['sent_tokens']:
        tweet_polarity = 0
        for sent in tweet:
            sentiment = sia.polarity_scores(sent)
            tweet_polarity += sentiment['compound']
        Vader_Sent_polarity.append(tweet_polarity)
    
    df['Vader_Sent_polarity'] = pd.Series(Vader_Sent_polarity)


    # analyse the tokenized words of each tweet
    Vader_Word_polarity = []

    for tweet in df['word_tokens']:
        tweet_polarity = 0
        for word in tweet:
            sentiment = sia.polarity_scores(word)
            tweet_polarity += sentiment['compound']
        Vader_Word_polarity.append(tweet_polarity)
    
    df['Vader_Word_polarity'] = pd.Series(Vader_Word_polarity)
        
    return df



"""analyse sentiment by modified Vader - taking tfidf into consideration"""

def modified_Vader_sentiment_analysis(df):

    sia = NLTK_Vader.SentimentIntensityAnalyzer()
    
    # analyse the raw text of each tweet
    Vader_Text_tfidf_polarity = []
    
    for tweet in df['Text']:

        tfidf_dict = df[df['Text']==tweet]['tfidf'].tolist()[0]
        
        sentiment = sia.polarity_scores(tweet, tfidf_dict)
        Vader_Text_tfidf_polarity.append(sentiment['compound'])
    
    df['Vader_Text_tfidf_polarity'] = pd.Series(Vader_Text_tfidf_polarity)
    
    
    # analyse the processed text of each tweet
    Vader_processed_text_tfidf_polarity = []
    
    for tweet in df['processed_text']:

        tfidf_dict = df[df['processed_text']==tweet]['tfidf'].tolist()[0]
        
        sentiment = sia.polarity_scores(tweet, tfidf_dict)
        Vader_processed_text_tfidf_polarity.append(sentiment['compound'])
    
    df['Vader_processed_text_tfidf_polarity'] = pd.Series(Vader_processed_text_tfidf_polarity)
    
    
    # analyse the tokenized sentences in each tweet
    Vader_Sent_tfidf_polarity = []
    
    for tweet in df['Text']:
        
        tfidf_dict = df[df['Text']==tweet]['tfidf'].tolist()[0]
        
        tweet_polarity = 0
        for sent in df[df['Text']==tweet]['sent_tokens'].tolist()[0]:
            sentiment = sia.polarity_scores(sent, tfidf_dict)
            tweet_polarity += sentiment['compound']       
        Vader_Sent_tfidf_polarity.append(tweet_polarity)

    df['Vader_Sent_tfidf_polarity'] = pd.Series(Vader_Sent_tfidf_polarity)
    
    return df




"""analyse sentiment by TextBlob"""

def Textblob_sentiment_analysis(df):
        
    # analyse the Text of each tweet
    TextBlob_Text_polarity = []
    
    for tweet in df['Text']:
        sentiment = TextBlob(tweet).sentiment
        TextBlob_Text_polarity.append(sentiment.polarity)
    
    df['TextBlob_Text_polarity'] = pd.Series(TextBlob_Text_polarity)
    
    
    # analyse the processed text of each tweet
    TextBlob_processed_text_polarity = []
    
    for tweet in df['processed_text']:
        sentiment = TextBlob(tweet).sentiment
        TextBlob_processed_text_polarity.append(sentiment.polarity)
    
    df['TextBlob_processed_text_polarity'] = pd.Series(TextBlob_processed_text_polarity)

    # analyse the tokenized sentences in each tweet
    TextBlob_Sent_polarity = []
    
    for tweet in df['sent_tokens']:
        tweet_polarity = 0
        for sent in tweet:
            sentiment = TextBlob(sent).sentiment
            tweet_polarity += sentiment.polarity
        TextBlob_Sent_polarity.append(tweet_polarity)
        
    df['TextBlob_Sent_polarity'] = pd.Series(TextBlob_Sent_polarity)
    
    
    # analyse the tokenized words of each tweet
    TextBlob_Word_polarity = []

    for tweet in df['word_tokens']:
        tweet_polarity = 0
        for word in tweet:
            sentiment = TextBlob(word).sentiment
            tweet_polarity += sentiment.polarity
        TextBlob_Word_polarity.append(tweet_polarity)
    
    df['TextBlob_Word_polarity'] = pd.Series(TextBlob_Word_polarity)
    
    return df
    
    
def accuracy_checking(df, column_x, column_y):
    
    x = df[column_x]
    y = df[column_y]
    length = len(df)
    accurate_number = 0
    
    for i in range(length):
        if (float(x[i]) > 0) & (float(y[i]) > 0):
            accurate_number += 1
        elif (float(x[i]) < 0) & (float(y[i]) < 0):
            accurate_number += 1
        elif (float(x[i]) == 0) & (float(y[i]) == 0):
            accurate_number += 1

    
    accuracy = accurate_number/length
    
    return accuracy
    
        
    
# '''testing results'''

# # NLTK Vader
# NV_positive = len(df.query("NLTK_Vader_polarity > 0"))
# NV_negative = len(df.query("NLTK_Vader_polarity < 0"))
# NV_neutral = len(df)-NV_positive-NV_negative
# # TextBlob
# TB_positive = len(df.query("TextBlob_polarity > 0"))
# TB_negative = len(df.query("TextBlob_polarity < 0"))
# TB_neutral = len(df)-TB_positive-TB_negative

# plt.subplots()
# plt.bar(['NV_positive','NV_negative','NV_neutral'],[NV_positive,NV_negative,NV_neutral])
# plt.bar(['TB_positive','TB_negative','TB_neutral'],[TB_positive,TB_negative,TB_neutral])
# for index,data in enumerate([NV_positive,NV_negative,NV_neutral,TB_positive,TB_negative,TB_neutral]):
#     plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=11), ha='center')
# plt.xticks(rotation='vertical')
# plt.title('Distribution of Polarity Scores Class')
# plt.ylabel('Number of Tweets')


# # Comparison
# df_NVp_TBp = df.query("NLTK_Vader_polarity > 0 and TextBlob_polarity > 0")
# df_NVn_TBn = df.query("NLTK_Vader_polarity < 0 and TextBlob_polarity < 0")
# df_NV0_TB0 = df.query("NLTK_Vader_polarity == 0 and TextBlob_polarity == 0")
# df_NVp_TBn = df.query("NLTK_Vader_polarity > 0 and TextBlob_polarity < 0")
# df_NVn_TBp = df.query("NLTK_Vader_polarity < 0 and TextBlob_polarity > 0")
# df_NVp_TB0 = df.query("NLTK_Vader_polarity > 0 and TextBlob_polarity == 0")
# df_NVn_TB0 = df.query("NLTK_Vader_polarity < 0 and TextBlob_polarity == 0")
# df_NV0_TBp = df.query("NLTK_Vader_polarity == 0 and TextBlob_polarity > 0")
# df_NV0_TBn = df.query("NLTK_Vader_polarity == 0 and TextBlob_polarity < 0")


# NVp_TBp = len(df_NVp_TBp)
# NVn_TBn = len(df_NVn_TBn)
# NV0_TB0 = len(df_NV0_TB0)
# NVp_TBn = len(df_NVp_TBn)
# NVn_TBp = len(df_NVn_TBp)
# NVp_TB0 = len(df_NVp_TB0)
# NVn_TB0 = len(df_NVn_TB0)
# NV0_TBp = len(df_NV0_TBp)
# NV0_TBn = len(df_NV0_TBn)


# plt.subplots()
# index = ['NVp_TBp','NVn_TBn','NV0_TB0','NVp_TBn','NVn_TBp','NVp_TB0','NVn_TB0','NV0_TBp','NV0_TBn']
# value = [NVp_TBp, NVn_TBn, NV0_TB0, NVp_TBn, NVn_TBp, NVp_TB0, NVn_TB0, NV0_TBp, NV0_TBn]
# colors = ['g','g','g','r','r','r','r','r','r']
# plt.bar(index, value, color=colors)
# for index,data in enumerate([NVp_TBp, NVn_TBn, NV0_TB0, NVp_TBn, NVn_TBp, NVp_TB0, NVn_TB0, NV0_TBp, NV0_TBn]):
#     plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=11), ha='center')
# plt.xticks(rotation='vertical')
# plt.title('The Difference of Polarity Score Class judged by NLTK Vader and TextBlob')
# plt.ylabel('Number of Tweets')




# df_diff1 = df.query("NLTK_Vader_polarity > 0 and TextBlob_polarity <= 0")
# df_diff2 = df.query("NLTK_Vader_polarity < 0 and TextBlob_polarity >= 0")
# df_diff3 = df.query("NLTK_Vader_polarity == 0 and TextBlob_polarity != 0")
# df_diff = pd.concat([df_diff1,df_diff2,df_diff3])
# # df_diff.to_csv('../dataset/fusun-pharma-2020-twitter-dataset/sentiment_testing_result.csv')


# # check accuracy

# df_diff_check = pd.read_excel('../dataset/fusun-pharma-2020-twitter-dataset/sentiment_testing_result.xlsx')

# df_NV1 = df_diff_check.query("NLTK_Vader_polarity > 0 and sentiment_manually > 0")
# df_NV2 = df_diff_check.query("NLTK_Vader_polarity < 0 and sentiment_manually < 0")
# df_NV3 = df_diff_check.query("NLTK_Vader_polarity == 0 and sentiment_manually == 0")
# df_NV = pd.concat([df_NV1,df_NV2,df_NV3])
# NV_accuracy = (len(df_NV)+len(df_NVp_TBp)+len(df_NVn_TBn)+len(df_NV0_TB0))/len(df)*100
# print("The accuracy of NLTK Vader polarity: %.2f%%" % NV_accuracy)


# df_TB1 = df_diff_check.query("TextBlob_polarity > 0 and sentiment_manually > 0")
# df_TB2 = df_diff_check.query("TextBlob_polarity < 0 and sentiment_manually < 0")
# df_TB3 = df_diff_check.query("TextBlob_polarity == 0 and sentiment_manually == 0")
# df_TB = pd.concat([df_TB1,df_TB2,df_TB3])
# TB_accuracy = (len(df_TB)+len(df_NVp_TBp)+len(df_NVn_TBn)+len(df_NV0_TB0))/len(df)*100
# print("The accuracy of TextBlob Vader polarity: %.2f%%" % TB_accuracy)