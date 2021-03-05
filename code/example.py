import data_reshaping_and_preprocessing as drp
import sentiment_analysis as sa
import regression_with_sentiment_polarity_score as Rrp
import pandas as pd

# Dataframe with columns - UserScreenName,UserName,Timestamp,Text,Emojis,Comments,Likes,Retweets,Image link,Tweet URL
data_frame = drp.read_file("../dataset/fusun-pharma-2020-twitter-dataset/fosun_pharma_2020.csv")

# Column - word_tokens and sent_tokens added to dataframe
data_frame = drp.tokenize_and_add_column(data_frame)
# Column - tfdif added to dataframe 
data_frame = drp.calculate_tfidf_and_add_column(data_frame)
# Column - NLTK_Vader_polarity and its score are added to dataframe
data_frame = sa.NLTK_Vader_sentiment_analysis(data_frame)


data_frame.to_csv("../dataset/test.csv")

# # Column - TextBlob_polarity added to dataframe
# data_frame = sa.TextBlob_sentiment_analysis(data_frame)
# # Calculate the daily polarity score
# df_polarity = Rrp.polarity_calculation(data_frame)


# # Dataframe with daily stock price - Data, Open, High, Low, Close, Adj Close, Column
# df_stock = pd.read_csv("../dataset/fusun-pharma-2020-stock-dataset/600196.csv")
# # Calculate stock daily returns and cumulative returns
# df_stock = Rrp.return_calculation(df_stock)


# # different regression models
# df_test = pd.merge(df_stock, df_polarity, how='left', on=['Date'] )
# Rrp.graphical_regression(df_test)
# Rrp.OLS_regression(df_test)


