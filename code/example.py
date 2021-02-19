import data_reshaping_and_preprocessing as drp

# Dataframe with columns - UserScreenName,UserName,Timestamp,Text,Emojis,Comments,Likes,Retweets,Image link,Tweet URL
data_frame = drp.read_file("./dataset/fusun-pharma-2020-twitter-dataset/fosun_pharma_2020.csv")
# Column - final_token added to dataframe
data_frame = drp.tokenize_and_add_column(data_frame)
# Column - tfdif added to dataframe 
data_frame = drp.calculate_tfidf_and_add_column(data_frame)
