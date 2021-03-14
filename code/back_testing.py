import pandas as pd

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
        elif df.loc[i,'Vader_Text_polarity'] < sentiment_low:
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
bank_acc = 0
invest_total = 0
cashflow_discount = 0
for i in range(len(df_backtesting)):
    
   
    if df_backtesting['Trade'].iloc[i] == 'buy':
        bank_acc = bank_acc - 1
        stock_acc = 1
        cashflow_discount = cashflow_discount - 1/((1+0.0137%)**i)
        for j in range(i+1,len(df_backtesting)):
            if df_backtesting['Trade'].iloc[j] == 'sell' or j == len(df_backtesting): 
                stock_acc_new = stock_acc*(df_backtesting['Adj Close'].iloc[j]/df_backtesting['Adj Close'].iloc[i])
                bank_acc = bank_acc + stock_acc_new
                cashflow_discount = cashflow_discount + stock_acc_new/((1+0.0137%)**i)
                break   
           # 遇到第一个sell就全部卖掉，这笔做多就结束了

    # elif df_backtesting['Trade'].iloc[i] == 'sell':
    #     # we don't need to set stock_acc, because investor will borrow stock and sell them
    #     bank_acc = bank_acc + 1  #sell stock and receive moneny
    #     for j in range(i+1, len(df_backtesting)):
    #         if df_backtesting['Trade'].iloc[j] == 'buy': 
    #             bank_acc = bank_acc-1*(df_backtesting['Adj Close'].iloc[j]/df_backtesting['Adj Close'].iloc[i])
    #             break # 遇到第一个buy就全部赎回，这笔做空就结束了
    #     # still don't need to set stock_acc, because investor will buy stock and return them

print(bank_acc)
# print(invest_total)

