# Data Visulization 
Followed by Data Reshaping and Preprocessing, we obtained a series of tokenized words. Due to the large amount of data set, it is hard for us to uderstand the underlying trends and patterns. Therefore, the data visulization becomes neccesary. As a part of data visulization, the frequency table and wordclouds are genenarated to tell the story behind the data.   

## 1. Process
### 1.1 Generating Frequency Set 
Firstly, we use dictionary in Python to count the frequencies in the 'final_token' list obtained from data shaping. and increment the counter using loops. 
```python
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

frequency = dict()

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
```
Note:The Def in python is short for "define", which performs a specific task. Therefore, the statements under Def should run together. 

### 1.2 Word Clouds
Word clouds  are visual representations of words. Here our geoup use word clouds to popular words based on word frequency. In this way, the large amount of data could be present in a quick and simple visual insights. 
```python 
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
```
Note: To use wordcloud, we need to install worldcloud package from 
      - git clone https://github.com/amueller/word_cloud.git\
      - cd word_cloud\
      - pip install\
      - OR easy downland with conda install -c conda-forge wordcloud

### 1.3 Frequency Table 

The frequency table shows the number of times a word occurring in Fosun tweets. We stored the frequency value key:value pairs and used a one-dimensional array (pd.series) to store the data. 

```python
final_frequency ={k:v for (k,v) in frequency.items() if not k in stopwords}

frequency_list = pd.Series(final_frequency).sort_values(ascending = False).head(30).to_frame().reset_index().rename(columns={'index':'word',0:'frequency'})

frequency_list.plot(kind='barh',x='word',y='frequency',figsize=(10,10))

plt.title('frequency of differnt words in Fosun tweets')

plt.show()
```
## 2. Results 
The world could picture and frequency table generating from the frequency of words in Fosun tweets are shown as below:

![Screenshot 2021-02-21 at 09 24 59](https://user-images.githubusercontent.com/78474798/108612799-e2f1aa80-7426-11eb-889e-844d6d445c7d.png)

## 3. Problems 

After finishing data visulization, we found two unrecoginized words appearing in our frequency table and world cloud of Fosun. Therefore, we re-examineed previous process step by step and ran the frequency of words in Bilibili tweets to do the rain check. The tables presenting the frequency of words in Bilibili tweets contains unregonized words as well. By checking the 'Variable Explorer' in Python, we discovered that these two unrecognized words are actually 'Chinese'. It shows the potential problems in the previous step, Data Reshaping and Preprocessing. It is always good to detect the problems in the early stage as it gives us the opportunity to correct the mistakes and stop loss in time. The process of analyzing natural language data would never be a smooth sailing and the most important thing is to keep going and moving forward. 

