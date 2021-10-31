import pandas as pd 
import matplotlib.pyplot as plt
from pandas.core import frame 

pd.options.mode.chained_assignment = None


# number last check 
n_last_price = 200
# number feature
n_feature = 19    #10 looop
#point_entry
poit_entry = 30
#point_exit
poit_exit = 40

def RSIcalc(asset):
    dataset = pd.read_csv("data.csv")
    dataset['MA200'] = dataset['close'].rolling(window=n_last_price).mean()
    dataset['price change'] = dataset['close'].pct_change()
    dataset['Upmove'] = dataset['price change'].apply(lambda x: x if x > 0 else 0)
    dataset['Downmove'] = dataset['price change'].apply(lambda x: abs(x) if x < 0 else 0)
    dataset['avg Up'] = dataset['Upmove'].ewm(span=n_feature).mean()
    dataset['avg Down'] = dataset['Downmove'].ewm(span=n_feature).mean()
    dataset = dataset.dropna()
    dataset['RS'] = dataset['avg Up']/dataset['avg Down']
    dataset['RSI'] = dataset['RS'].apply(lambda x: 100-(100/(x+1)))
    dataset.loc[(dataset['close'] > dataset['MA200']) & (dataset['RSI'] < poit_entry),'Buy'] = 'Yes'
    dataset.loc[(dataset['close'] < dataset['MA200']) | (dataset['RSI'] > poit_entry),'Buy'] = 'No'
    print(dataset)
    return dataset


def getSingnal(dataset):
    Buying_dates = []
    Selling_dates = []

    for i in range(len(dataset)):
        if 'Yes' in dataset['Buy'].iloc[i]:
            Buying_dates.append(dataset['close'].iloc[i+1])
            Selling_dates.append(float('nan'))
            for j in range(1,11):
                if dataset['RSI'].iloc[i + j] > poit_exit:
                    Selling_dates.append(dataset['close'].iloc[i+j+1])
                    Buying_dates.append(float('nan'))
                    break
                elif j == 10:
                    Selling_dates.append(dataset['close'].iloc[i+j+1])
                    Buying_dates.append(float('nan'))
                    
    return Buying_dates, Selling_dates







if __name__ == '__main__':
    dataset = RSIcalc("BTC")
    buy, sell = getSingnal(dataset)
    print(buy)
    dataset['buy_signals'] = buy
    dataset['sell_signals'] = sell

    plt.figure(figsize=(19,6))
    plt.plot(dataset['close'] , label= "Share Price" , color = "lightgray")
    plt.scatter(dataset.index , dataset['buy_signals'] , label= "buy_signals" , marker="^" , color = "#00ff00" , lw = 3)
    plt.scatter(dataset.index , dataset['sell_signals'] , label= "sell_signals" , marker="v" , color = "#ff0000" , lw = 3)
    plt.legend(loc="upper left")
    plt.show()