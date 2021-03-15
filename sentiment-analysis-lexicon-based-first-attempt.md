# Lexicon-based approaches: First attempt

In the lexicon-based method, a polarity score will be assigned to the unigram which is found in the lexicon library and the overall polarity score is then computed by summing the polarities of the unigrams for longer text. (Kolchyna et al., 2015)

From data preprocessing, we removed urls, @ references and '#' from tweets, filtered out non-english words and stop wordstweets, and combined the different grammatical forms of the same words by implementing stemming and lemmatization. The next stage is sentiment analysis.

NLTK Vader and TextBlob are the most popular tools used to analyse sentiment from text. At this stage, we apply both of these two tools for our tokenized words which are listed at the `word_tokens` column of our twitter dataframe.


## Approach 1: NLTK Vader

### 1.1 Code

```python
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
    
NLTK_Vader_polarity = []

for line in twitter_data_frame['word_tokens']:
    sentiment = sia.polarity_scores(line)
    NLTK_Vader_polarity.append(sentiment)

twitter_data_frame['NLTK_Vader_polarity'] = pd.Series(NLTK_Vader_polarity)
```

### 1.2 Attempt Result

Before applying to the `word_tokens` column, this tool is used to analyse the sentiment of the `Text` column which contains raw text of each tweet for the purpose of testing the sensitivity. Different polarity scores are produced for different tweets and the results seem to be reasonable. 

With the confidence of application of this tool, we test the `word_tokens` column. However, instead of producing different polarity scores, it shows that all the tokenized words are neutral. 

<div align=center><img width = '800' height ='200' src ="./sentiment-testing/first attempt_vader.png"/></div>


## Approach 2: TextBlob

### 2.1 Code

```python
from textblob import TextBlob
TextBlob_polarity = []

for line in twitter_data_frame['word_tokens']:
    [polarity, subjectivity] = list(TextBlob(line).sentiment)
    TextBlob_polarity.append(polarity)
    
twitter_data_frame['TextBlob_polarity'] = pd.Series(TextBlob_polarity)
```

### 2.2 Attempt Result

The same procedure as before, this tool is used to analyse sentiment of the raw tweet text before applying to the `word_tokens` column. It produces the corresponding polarity scores for individual tweets, but the results are very different from that obtained by *NLTK Vader*.

While we are applying this tool to the `word_tokens` column, it is also sensitive to our tokenized word sets. We proceed the regression procedure with these results at this stage.


## 3. Things to Be Improved

- Find out why *NLTK Vader* is not useful when analysing the tokenized words
- Since the polarity scores produced by *NLTK Vader* and *TextBlob* are very different, identify the difference of the two methods.
- Find out how to apply the TFIDF weights for sentiment analysis.


## Reference

1. Kolchyna, O., Souza, T., Treleaven, P., Aste, T. (2015) , *Twitter Sentiment Analysis: Lexicon Method, Machine Learning Method and Their Combination*, viewed 12 March 2011, <https://arxiv.org/abs/1507.00955>.