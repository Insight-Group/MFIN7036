# Lexicon-based approaches: First attempt

From previous data processing, we filtered english tweets and tokenized each tweet by using meaningful words. 

NLTK Vader and TextBlob are the most popular tools used to analyse sentiment from text. At this stage, we apply both of these two tools for our tokenized words which are listed at the `final_token` column of our twitter dataframe.


## Approach 1: NLTK Vader

### 1.1 Code

```python
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
    
NLTK_Vader_polarity = []

for line in twitter_data_frame['final_token']:
    sentiment = sia.polarity_scores(line)
    NLTK_Vader_polarity.append(sentiment)

twitter_data_frame['NLTK_Vader_polarity'] = pd.Series(NLTK_Vader_polarity)
```

### 1.2 Attempt Result

Before applying to the `final_token` column, this tool is used to analyse the sentiment of the `Text` column which contains all the raw text of each tweet, for the purpose of testing the sensitivity. Different polarity scores are produced for different tweets and the results seem to be reasonable. 

With the confidence of application of this tool, we test the `final_token` column. However, instead of producing different polarity scores, it shows that all tokenized word sets are neutral. 


## Approach 2: TextBlob

### 2.1 Code

```python
from textblob import TextBlob
TextBlob_polarity = []

for line in twitter_data_frame['final_token']:
    [polarity, subjectivity] = list(TextBlob(line).sentiment)
    TextBlob_polarity.append(polarity)
    
twitter_data_frame['TextBlob_polarity'] = pd.Series(TextBlob_polarity)
```

### 2.2 Attempt Result

The same procedure as before, this tool is used to analyse sentiment of the raw tweet text before applying to the `final_token` column. It produces the corresponding polarity scores for individual tweets, but the results are very different from that obtained by *NLTK Vader*.

While we are applying this tool to the `final_token` column, it is also sensitive to our tokenized word sets. We proceed the regression procedure with these results at this stage.


## 3. Things to Be Improved

- Find out why *NLTK Vader* is not useful when analysing the tokenized words
- Since the polarity scores produced by *NLTK Vader* and *TextBlob* are very different, identify the difference of the two methods.
- Find out how to apply the TFIDF weights for sentiment analysis.