# What is back testing?
- To testify the real existance of correlation between sentiment and return, we want to utilize our strategy to invest based on historical financial data

# Regression result to base on
- In our daily regression result, we find that daily sentiment has positive correlation with next day's stock return. We want to bring such discovery to investment. If we observe that today's sentiment is high enough, we will buy shares. However, if we observe that today's sentiment is low enough, we will sell our shares and further short our shares.

## Question 1: How to judge if the sentiment is high or low enough?
- This is the area we need to try again and again. As the sentiment will change its scale much more if the period is so long, we select data according to monthly rolling basis. If today's sentiment is over 75% quantile of the past 30 days, we view it as a high point. If today's sentiment is under 25% quantile of the past 30 days, we view it as a low point. Maybe '30 days' and '75%/25%' are not good standards, but this is just a good perspective, we could try more and more to find suitable standards.

- Code:

```python
# Sentiment reach 75% high in recent 1 month, we view this as buying signal.
# Sentiment reach 25% low in recent 1 month, we view this as selling signal.
def judgement_point(df):
    Trade = []
    for i in range(30,len(df)):

        # settle the value which could be viewed as high or low
        sentiment_high = df.loc[i-30:i,'Vader_Text_polarity'].quantile(0.75)
        sentiment_low = df.loc[i-30:i,'Vader_Text_polarity'].quantile(0.25)
        
        if df.loc[i,'Vader_Text_polarity'] > sentiment_high:
            Trade.append('buy')
        elif df.loc[i,'Vader_Text_polarity'] < sentiment_low):
            Trade.append('sell')
        else:
            Trade.append('')
    # add sell and buy point into the dataframe         
    df['Trade'] = pd.Series(Trade)
    return df
```
## Question 2: Why buy shares and sell them until the next 'selling signal' and Why short shares and buy them back until the next 'buying signal'?
- You may ask me about the above question. We have considered such question: because we past 2 years are bull market, if we just buy and sell at several signals, we cannot compare our result with market. What we need to do is buy and sell at several important signals and just hold them in the spare time so that we can compare the result with buy-and-hold' result. In this way, our strategy could be more prersuading.

# Backtesting
- Logic: we have bank account and stock account. Everytime we long stocks, we borrow money from the bank, buy shares and our stock account would increase. If we sell them, the change is reversed. Everytime we short stocks, we borrow stocks and sell them, so the bank account would increase but the stock account would not change. If we buy back the stock and return them, the stock account still would not change but the bank account would decrease.

- we also count our investment and discount our cashflow to the first investment point. We compare the return from buy-and-hold strategy commencing at the same time.

- Code:

```python
df_bilibili_polarity_return = pd.read_csv("../dataset/bilibili_backtesting/bilibili_polarity_return.csv")
df_backtesting = judgement_point(df_bilibili_polarity_return).dropna()
df_backtesting.to_csv("../dataset/bilibili_backtesting/back_testing.csv")

# *****Back Testing*****
# for every single row, it needs to be judged whether to buy or sell

cashflow_in_discount = 0
cashflow_out_discount = 0
record = []
for i in range(len(df_backtesting)):
    record.append(i)
    start_point = record[0]
    if df_backtesting['Trade'].iloc[i] == 'buy':
        stock_acc = 1
        cashflow_out_discount += 1/((1+0.000137)**(i-start_point))
        for j in range(i+1,len(df_backtesting)):
            if df_backtesting['Trade'].iloc[j] == 'sell' or j == len(df_backtesting): 
                stock_acc_new = 1*(df_backtesting['Adj Close'].iloc[j]/df_backtesting['Adj Close'].iloc[i])
                cashflow_in_discount += stock_acc_new/((1+0.000137)**(j-start_point))
                break   
           # when encountering selling point, this round of buying stock would be over

    elif df_backtesting['Trade'].iloc[i] == 'sell':
        cashflow_in_discount += 1/((1+0.000137)**(i-start_point))
        for j in range(i+1, len(df_backtesting)):
            if df_backtesting['Trade'].iloc[j] == 'buy' or j == len(df_backtesting): 
                cashflow_out_discount += 1*(df_backtesting['Adj Close'].iloc[j]/df_backtesting['Adj Close'].iloc[i])/((1+0.000137)**(j-start_point))
                break 
        # when encountering buying point, this round of short selling stock would be over

return_at_first_point = (cashflow_in_discount - cashflow_out_discount)/cashflow_out_discount
print(return_at_first_point)
