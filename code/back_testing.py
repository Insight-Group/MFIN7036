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
