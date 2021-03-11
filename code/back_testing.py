import pandas as pd

#但是要记住，用过去一年的数值去找，找到基准之后用未来的一年去测
#可以规定sentiment大于75%，同时return小于25%为买点：
sentiment_high = df_daily.polarity_change(0.75)
return_low = df_daily.daily_return(0.25)
#可以规定sentiment小于25%，同时return大于75%为卖点：
sentiment_high = df_daily.polarity_change(0.25)
return_low = df_daily.daily_return(0.75)
#先找75% 和 25%的基准在哪里？


df_daily = pd.read_csv('../dataset/test_backtesting.csv')

# daily analysis
# 对于每一列 都需要让它自己跑完：包含判断自己是buy还是sell， 什么时候进行账户清算
bank_acc = 0
stock_acc = 0
invest_total = 0
for i in range(len(df_daily.index)):
    if df_daily.loc[i, 'Trade'] == 'buy':
        bank_acc = bank_acc - 1
        invest_total = invest_total + 1  #/时间价值*****
        stock_acc = stock_acc + 1
        for j in range(i+1,len(df_daily.index)):
            if df_daily.loc[j, 'Trade'] == 'sell':
                stock_acc = stock_acc * (df_daily.loc[j,'Adj Close']/df_daily.loc[i,'Adj Close'])
                bank_acc = bank_acc + stock_acc
                stock_acc = 0 
                break # 遇到第一个sell就全部卖掉，这笔做多就结束了

    elif df_daily.loc[i, 'Trade'] == 'sell':
        stock_acc = 0 # borrow stock and sell them
        bank_acc = bank_acc + 1 # sell stock and receive moneny
        for j in range(i+1, len(df_daily.index)):
            if df_daily.loc[j, 'Trade'] == 'buy':
                stock_acc = 0 # buy stock and return them
                bank_acc = bank_acc - 1*(df_daily.loc[j,'Adj Close']/df_daily.loc[i,'Adj Close'])
                break # 遇到第一个buy就全部赎回，这笔做空就结束了

print(stock_acc)
print(invest_total)
print(bank_acc)
