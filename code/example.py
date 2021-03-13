import pandas as pd
import data_reshaping_and_preprocessing as drp
import data_visualization as dv
import sentiment_analysis_Vader_and_TextBlob as sa_vdtb
import sentiment_analysis_Machine_Learning_Models as sa_ml
import time as t
import regression_strategy as reg


start_time = t.time()  # check the running time

# Data loading=================================================================

print('\nReading data...\n')
# Read data downloaded by Scweet(https://github.com/Altimis/Scweet). The csv file contains columns - UserScreenName,UserName,Timestamp,Text,Emojis,Comments,Likes,Retweets,Image link,Tweet URL
df_twitter = pd.read_csv("../dataset/bilibili-2018-2021-twitter-dataset/bilibili_2018-01-01_2021-02-09.csv")
print('\nPrint example tweets after reading the csv file')
print(df_twitter.head())

# Read data downloaded from Yahoo Finance. The csv file contains columns - Date,Open,High,Low,Close,Adj Close,Volume
df_stock = pd.read_csv("../dataset/bilibili-2018-2021-stock-dataset/BILI.csv", index_col=False)
print('\nPrint example prices after reading the csv file')
print(df_stock.head())

# Reference dataset which has been manually judged by us (MUST contain a column called 'sentiment_manually')
df_reference = pd.read_excel("../dataset/fusun-pharma-2020-twitter-dataset/sentiment_manually checking.xlsx")  # we have manually checked the dataset of fusun pharma

# Training dataset (MUST contain a column called 'sentiment_manually')
df_training_dataset = pd.read_csv("../dataset/training dataset/training_dataset.csv")           # Sentiment140 dataset (1.6 millions tweets), too big to run
training_size = input('How big training size do you expect? (i.e. how many trained tweets): ')  
df_training = df_training_dataset.sample(n = int(training_size)).reset_index()                  # shrink the size of the training dataset


# Twitter data processing======================================================

print('\nProcessing data...')

# Add Columns - processed_text, word_tokens and sent_tokens
df_twitter = drp.tokenize_and_add_column(df_twitter)
df_training = drp.tokenize_and_add_column(df_training)

# Add Column - tfdif
df_twitter = drp.calculate_tfidf_and_add_column(df_twitter)

# Data visualization
# dv.data_visualization(df_twitter['word_tokens'])


# Sentiment analysis-==========================================================

# Lexicon-based approaches-----------------------------------------------------

print('\nAnalysing the sentiment of data by lexicon-based approaches...')

# Add Columns - use Vader to analyze the sentiment of tweets based on raw text, sent_tokens, word_tokens
df_twitter = sa_vdtb.NLTK_Vader_sentiment_analysis(df_twitter)

# Add Columns - modify the valences of words in Vader by tfidf weighting and 
#               use the modified Vader to analyze the sentiment of tweets based on raw text, sent_tokens
df_twitter = sa_vdtb.modified_Vader_sentiment_analysis(df_twitter)

# Add Columns - use TextBlob to analyze the sentiment of tweets based on raw text, sent_tokens, word_tokens
df_twitter = sa_vdtb.Textblob_sentiment_analysis(df_twitter)


# Machine learning approaches--------------------------------------------------
print('\nAnalysing the sentiment of data by machine learning approaches...')

# Add column - anaylzed results by Naive Bayes, 'prediction_nb' 
df_twitter['prediction_nb'] = pd.Series(sa_ml.Naive_Bayes_model(df_twitter['processed_text'], 
                                                                df_training['processed_text'], 
                                                                df_training['sentiment_manually']))

# Add column - anaylzed results by Logistics Regression, 'prediction_lr' 
df_twitter['prediction_lr'] = pd.Series(sa_ml.Logistics_Regression_model(df_twitter['processed_text'],
                                                                         df_training['processed_text'],
                                                                         df_training['sentiment_manually']))

# Add column - anaylzed results by SVM, 'prediction_svc' 
df_twitter['prediction_svc'] = pd.Series(sa_ml.Support_Vector_Machines(df_twitter['processed_text'],
                                                                       df_training['processed_text'],
                                                                       df_training['sentiment_manually']))

# # Checking the accuracies according to the reference dataframe-----------------

# df_twitter = pd.concat([df_twitter, df_reference['sentiment_manually']], axis=1)

# print('\nPrinting the accuracy of each approach...\n')
# # Vader
# print('Accuracy of Vader based on raw text is: ', sa_vdtb.accuracy_checking(df_twitter, 'Vader_Text_polarity', 'sentiment_manually'))
# print('Accuracy of Vader based on processed text is: ', sa_vdtb.accuracy_checking(df_twitter, 'Vader_processed_text_polarity', 'sentiment_manually'))
# print('Accuracy of Vader based on sent_tokens is: ', sa_vdtb.accuracy_checking(df_twitter, 'Vader_Sent_polarity', 'sentiment_manually'))
# print('Accuracy of Vader based on word_tokens is: ', sa_vdtb.accuracy_checking(df_twitter, 'Vader_Word_polarity', 'sentiment_manually'))
# print('Accuracy of modified Vader based on raw text is: ', sa_vdtb.accuracy_checking(df_twitter, 'Vader_Text_tfidf_polarity', 'sentiment_manually'))
# print('Accuracy of modified Vader based on processed text is: ', sa_vdtb.accuracy_checking(df_twitter, 'Vader_processed_text_tfidf_polarity', 'sentiment_manually'))
# print('Accuracy of modified Vader based on sent_tokens is: ', sa_vdtb.accuracy_checking(df_twitter, 'Vader_Sent_tfidf_polarity', 'sentiment_manually'))
# # TextBlob
# print('Accuracy of TextBlob based on raw text is: ', sa_vdtb.accuracy_checking(df_twitter, 'TextBlob_Text_polarity', 'sentiment_manually'))
# print('Accuracy of TextBlob based on processed text is: ', sa_vdtb.accuracy_checking(df_twitter, 'TextBlob_processed_text_polarity', 'sentiment_manually'))
# print('Accuracy of TextBlob based on sent_tokens is: ', sa_vdtb.accuracy_checking(df_twitter, 'TextBlob_Sent_polarity', 'sentiment_manually'))
# print('Accuracy of TextBlob based on word_tokens is: ', sa_vdtb.accuracy_checking(df_twitter, 'TextBlob_Word_polarity', 'sentiment_manually'))
# # Naive Bayes Model
# print('Accuracy of Naive_Bayes_model is: ', sa_vdtb.accuracy_checking(df_twitter, 'prediction_nb', 'sentiment_manually'))
# # Logistics Regression Model
# print('Accuracy of Logistics_Regression_model is: ', sa_vdtb.accuracy_checking(df_twitter, 'prediction_lr', 'sentiment_manually'))
# # SVM Model
# print('Accuracy of Support_Vector_Machines: ', sa_vdtb.accuracy_checking(df_twitter, 'prediction_svc', 'sentiment_manually'))

df_twitter.to_csv('../dataset/test.csv')



# Regression strategies========================================================

   
# select polarity results from different sentiment analysis approaches
choice = input('Use polarity results from which approach? [Options: lexicon-based approach, machine learning approach]\n')
               
best_sentiment_analysis_results = {'lexicon-based approach':    'Vader_Text_polarity',
                                   'machine learning approach': 'prediction_svc'} 
                  
polarity_column = best_sentiment_analysis_results[choice]

    
trading_timezone = 'America/New_York'    # for bilibili
# trading_timezone = 'Asia/Shanghai'     # for fusun pharma

trading_timeshift = +8                   # for HK stock market, US stock market
# trading_timeshift = +9                 # for Mainland China
 

reg.regression_strategy(df_stock, df_twitter, polarity_column, trading_timezone, trading_timeshift)

# =============================================================================

print('The total running time is %.2f seconds.' % (t.time() - start_time))