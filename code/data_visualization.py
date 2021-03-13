
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud



def word_count(df_column):
       
    frequency = dict()
    
    for line in df_column:
        for word in line:
            if word in frequency:
                frequency[word]+=1
            else:
                frequency[word]=1
                
    print(frequency)
    
    return frequency
    


def data_visualization(df_column):

    frequency = word_count(df_column)
    
    
    # Remove the unwanted special words which are not filtered out from the data preprocessing procedure
    stopwords_1 = nltk.corpus.stopwords.words('english')
    newStopWords = ('Fosun','Bilibili','Luckin','da','e','en','la','http','Pharma','el','de','pharma','Pharma','FosunPharma','para','un','le')
    stopwords_1.extend(newStopWords)
    print(stopwords_1)
    
    #worldcloud 
    #need to install worldcloud package from 
    #git clone https://github.com/amueller/word_cloud.git
    #cd word_cloud
    #pip install
    #or easy downland with conda install -c conda-forge wordcloud
       
    text = " ".join([(k+" ")*v for k,v in frequency.items()])
        
    wc = WordCloud(stopwords = stopwords_1, background_color="white", width = 30000, height = 20000,collocations = False,max_words=50)
    wc.generate(text)

    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.show


    # frequency table 

    final_frequency ={k:v for (k,v) in frequency.items() if not k in stopwords_1}
    
    frequency_list = pd.Series(final_frequency).sort_values(ascending = False).head(30).to_frame().reset_index().rename(columns={'index':'word',0:'frequency'})
    frequency_list.plot(kind='barh',x='word',y='frequency',figsize=(10,10))
    
    plt.title('frequency of differnt words in Fosun tweets')
    plt.show()









