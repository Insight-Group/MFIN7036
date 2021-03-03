# Sentiment Analysis: NLTK Vader or TextBlob?

From previous sentiment analysis on tweets of Fosun Pharma in 2020, the sentiment polarity of the same tweet judged by *NLTK Vader* and *TextBlob* is very different. Besides, *NLTK Vader* cannot recognize the sentiment of tokenized words. This blog discusses the main differences between these two sentiment analysis approaches and shows the accuracy of them based on the tweets of Fosun Pharma in 2020.

## 1 Main differences



## 2 Sentiment polarity scores

<div align=center><img width = '500' height ='350' src ="./sentiment-testing/Distribution of Polarity Scores Class.png"/></div>

Among 1123 tweets, the result given by *NLTK Vader* is that 390(34.73%), 142(12.64%) and 591(52.63%) are positive, negative and neutral respectively, while *TextBlob* says that 291(25.91%), 100(8.90%) and 732(65.18%) are positive, negative and neutral respectively. 

Both of these two approaches conclude that positive sentiment in tweets is greater than negative sentiment. Both of these two approaches conclude that positive sentiment in tweets is greater than negative sentiment. However, the numbers in each polarity scores class are not the same. More neutral are produced by *TextBlob*.

<div align=center><img width = '500' height ='350' src ="./sentiment-testing/The Difference of Polarity Score Class.png"/></div>

200 of the 1123 tweets are recognized to be positive by both of these two approaches, and 29 and 492 are recognized to be negative and neutral. These two approaches have different results on the polarity of the remaining 402 tweets. *NLTK Vader* gives positive and negative sentiment score for the 151 and 89 tweets, but *TextBlob* says they are neutral. So, *NLTK Vader* seems to be more sensitive to our sample tweets than *TextBlob*.

After scoring these 402 tweets manually and comparing the results given by these two approaches, *NLTK Vader* gives the same sentiment score to 181 of the tweets whereas *TextBlob* gives the same sentiment score to 139 of the tweets.

In overall, The accuracy of *NLTK Vader* in judging the sentiment polarity of these 1123 tweets is 80.32% and that of TextBlob is 76.58%.
