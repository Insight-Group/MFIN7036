import pandas as pd

# Sentiment reach 75% high in recent 1 month, we view this as buying signal.
# Sentiment reach 25% low in recent 1 month, we view this as selling signal.
def judgement_point(df):
    Trade = []
    for i in range(30,len(df)):

        # settle the value which could be viewed as high or low
        sentiment_high = df.loc[i-30:i,'Vader_Text_polarity'].quantile(0.75)
        sentiment_low = df.loc[i-30:i,'Vader_Text_polarity'].quantile(0.25)
        return_high = df.loc[i-30:i,'daily_return'].quantile(0.75)
        return_low = df.loc[i-30:i,'daily_return'].quantile(0.25)
        
        if ((df.loc[i,'Vader_Text_polarity'] > sentiment_high) & (df.loc[i,'daily_return'] < return_low)):
            Trade.append('buy')
        elif ((df.loc[i,'Vader_Text_polarity'] < sentiment_low) & (df.loc[i,'daily_return'] > return_high)):
            Trade.append('sell')
        else:
            Trade.append('')
    # add sell and buy point into the dataframe         
    df['Trade'] = pd.Series(Trade)
    return df

df_bilibili_polarity_return = pd.read_csv(r"D:\gitrepo\MFIN7036\dataset\bilibili_backtesting\bilibili_polarity_return.csv")
df_backtesting = judgement_point(df_bilibili_polarity_return).dropna()
df_backtesting.to_csv(r"D:\gitrepo\MFIN7036\dataset\bilibili_backtesting\back_testing.csv")



# *****Back Testing*****
# for every single row, it needs to be judged whether to buy or sell
bank_acc = 10000
invest_total = 0
stock_acc = 0
for i in range(len(df_backtesting)):

    if df_backtesting.loc[i, 'Trade'] == 'buy':
        bank_acc = bank_acc - 1
        invest_total = invest_total + 1  #/时间价值*****
        stock_acc = stock_acc + 1
        for j in range(i+1,len(df_backtesting)):
            if df_backtesting.loc[j, 'Trade'] == 'sell': 
                break   
            stock_acc = stock_acc*(df_backtesting.loc[j,'Adj Close']/df_backtesting.loc[j-1,'Adj Close'])

        stock_acc = stock_acc*(df_backtesting.loc[j,'Adj Close']/df_backtesting.loc[j-1,'Adj Close'])  
        bank_acc = bank_acc + stock_acc 
        stock_acc = 0
        # 遇到第一个sell就全部卖掉，这笔做多就结束了

    elif df_backtesting.loc[i, 'Trade'] == 'sell':
        # we don't need to set stock_acc, because investor will borrow stock and sell them
        bank_acc = bank_acc + 1  #sell stock and receive moneny
        for j in range(i+1, len(df_backtesting)):
            if df_backtesting.loc[j, 'Trade'] == 'buy': 
                break # 遇到第一个buy就全部赎回，这笔做空就结束了
            bank_acc = bank_acc*(df_backtesting.loc[j,'Adj Close']/df_backtesting.loc[j-1,'Adj Close'])

        bank_acc = bank_acc*(df_backtesting.loc[j,'Adj Close']/df_backtesting.loc[j-1,'Adj Close'])
        # still don't need to set stock_acc, because investor will buy stock and return them

print(bank_acc)
print(stock_acc)
print(invest_total)

