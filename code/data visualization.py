
import pandas as pd
import data_reshaping_and_preprocessing as drp

# Dataframe with columns - UserScreenName,UserName,Timestamp,Text,Emojis,Comments,Likes,Retweets,Image link,Tweet URL
data_frame = drp.read_file("/Users/luqilin/MFIN7036-Blog/dataset/fusun-pharma-2020-twitter-dataset/fosun_pharma_2020.csv")
# Column - final_token added to dataframe
data_frame = drp.tokenize_and_add_column(data_frame)



import matplotlib.pyplot as plt
from nltk.corpus import stopwords

frequency = dict()

#def as define function; need to run together 
df=pd.DataFrame(data_frame)
final_token=df.final_token

def word_count(final_token):
    for token in final_token:
        for word in token:
            if word in frequency:
                frequency[word]+=1
            else:
                frequency[word]=1
    return frequency
    

word_count(final_token)

print(frequency)


#worldcloud 
#need to install worldcloud package from 
#git clone https://github.com/amueller/word_cloud.git
#cd word_cloud
#pip install
#or easy downland with conda install -c conda-forge wordcloud

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import os
import numpy as np


text = " ".join([(k+" ")*v for k,v in frequency.items()])

stopwords = set(STOPWORDS)

path = "/Users/luqilin/MFIN7036-Blog/code/red-white-pill-hi.png"

pill = np.array(Image.open(path))

wc = WordCloud(stopwords = stopwords,mask = pill, background_color="white", width = 30000, height = 20000,collocations = False,max_words=50)

wc.generate(text)

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.show


#frequency table 


final_frequency ={k:v for (k,v) in frequency.items() if not k in stopwords}

frequency_list = pd.Series(final_frequency).sort_values(ascending = False).head(30).to_frame().reset_index().rename(columns={'index':'word',0:'frequency'})

frequency_list.plot(kind='barh',x='word',y='frequency',figsize=(10,10))

plt.title('frequency of differnt words in Fosun tweets')

plt.show()









