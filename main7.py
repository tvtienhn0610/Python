import datetime as dt
import pandas as pd
from matplotlib import pyplot as plt
from binance.client import Client
import datetime 
import os
import schedule
import time
import requests
import warnings
import numpy as np

plt.style.use("dark_background")

ma_1 = 50
ma_2 = 200


# number last check 
n_last_price = 200
# number feature
n_feature = 19    #10 looop
#point_entry
poit_entry = 30
#point_exit
poit_exit = 40

# high_signal_mfi
high_mfi = 81
low_mfi = 20

token_telegram = "2095464082:AAHw8loNeTSCtaoXgwBtpZdDz7vzIJW6MbY"
chatId_telegram = "1221651999"

# YOUR API KEYS HERE
api_key = "lmRl0bcG4my9tvoNDdgUCEuFKASmzwEPSUnvl4GIm3OUdNpdloQtGiTZcYioDen8"    #Enter your own API-key here
api_secret = "DgRJaDCnJvx5FJbx1qXbiduBbcLqtXxxjtxl360gC7RJnErlzQ5FnIC6hWa5Ydh3" #Enter your own API-secret here


def startProcess():
    try:
        print("start process tool every hour !!!")
        bclient = Client(api_key=api_key, api_secret=api_secret)
        print("start get all Name coin from binace !!!")
        tickers = bclient.get_all_tickers()
        for ticker in tickers:
            if str(ticker['symbol'])[-4:] == 'USDT':
                print(ticker['symbol'])
                processTool(ticker['symbol'],bclient)
        
    except:
        print("Eror start !!!")


def processTool(symbol,bclient):
    try:
        print("start process coint :"+symbol)  
        print("start down load data coin from 2020 !!!")
        start_date = datetime.datetime.strptime('1 Jun 2021', '%d %b %Y')
        today = datetime.datetime.now()
        
        klines = bclient.get_historical_klines(symbol, Client.KLINE_INTERVAL_2HOUR, start_date.strftime("%d %b %Y %H:%M:%S"), today.strftime("%d %b %Y %H:00:00"), 1000)
        data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

        data.set_index('timestamp', inplace=True)
        data.to_csv(symbol+".csv")
        print(data)
        # processGetBuySell(symbol)
        # processGetBuySellRSI(symbol)
        ProcessMFI(symbol)
        # processMACD(symbol)

    except:
        print("Error download data")  


def processGetBuySell( symblo):

    filename = symblo+".csv"
    dataset = dataset = pd.read_csv(filename)
    os.remove(filename)

    dataset[f'SMA_{ma_1}'] = dataset['close'].rolling(window=ma_1).mean()
    dataset[f'SMA_{ma_2}'] = dataset['close'].rolling(window=ma_2).mean()
    dataset = dataset.iloc[ma_2:]

    buy_signals = []
    sell_signals = []
    trigger = 0 

    for x in range(len(dataset)):
        if dataset[f'SMA_{ma_1}'].iloc[x] > dataset[f'SMA_{ma_2}'].iloc[x] and trigger != 1:
            buy_signals.append(dataset['close'].iloc[x])
            sell_signals.append(float('nan'))
            trigger = 1
        elif dataset[f'SMA_{ma_1}'].iloc[x] < dataset[f'SMA_{ma_2}'].iloc[x] and trigger != -1:
            buy_signals.append(float('nan'))
            sell_signals.append(dataset['close'].iloc[x])
            trigger = -1
        else:
            buy_signals.append(float('nan'))
            sell_signals.append(float('nan'))
    
    dataset['buy_signals'] = buy_signals
    dataset['sell_signals'] = sell_signals

    print(symblo , dataset['buy_signals'].iloc[-1] , dataset['sell_signals'].iloc[-1])
    buypoint = dataset['buy_signals'].iloc[-1]
    if buypoint > 0 :
        pusgMeassageTotele(symblo+" | buy | "+str(buypoint))
    
    sellpoint = dataset['sell_signals'].iloc[-1]
    if sellpoint > 0 :
        pusgMeassageTotele(symblo+" | sell | "+str(sellpoint))

    # plt.figure(figsize=(19,6))
    # plt.plot(dataset['close'] , label= "Share Price" , color = "lightgray")
    # plt.plot(dataset[f'SMA_{ma_1}'] , label= f"SMA_{ma_1}" , color = "orange", linestyle="--")
    # plt.plot(dataset[f'SMA_{ma_2}'] , label= f"SMA_{ma_2}" , color = "pink" , linestyle="--")
    # plt.scatter(dataset.index , dataset['buy_signals'] , label= "buy_signals" , marker="^" , color = "#00ff00" , lw = 3)
    # plt.scatter(dataset.index , dataset['sell_signals'] , label= "sell_signals" , marker="v" , color = "#ff0000" , lw = 3)
    # plt.legend(loc="upper left")
    # plt.show()



def processGetBuySellRSI( symblo):
    print("start 1")
    filename = symblo+".csv"
    dataset = dataset = pd.read_csv(filename)
    os.remove(filename)
    print("start 2")
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

    print("start 3")
    buy_signals = []
    sell_signals = []


    for i in range(len(dataset)):
            if 'Yes' in dataset['Buy'].iloc[i]:
                buy_signals.append(dataset['close'].iloc[i])
                sell_signals.append(float('nan'))
                # for j in range(1,11):
                #     if dataset['RSI'].iloc[i + j] > poit_exit:
                #         sell_signals.append(dataset['close'].iloc[i+j+1])
                #         buy_signals.append(float('nan'))
                #         break
                #     elif j == 10:
                #         sell_signals.append(dataset['close'].iloc[i+j+1])
                #         buy_signals.append(float('nan'))
                #     else :
                #         buy_signals.append(float('nan'))
                #         sell_signals.append(float('nan'))
            else:
                buy_signals.append(float('nan'))
                sell_signals.append(float('nan'))

    print("start 4")
    print(len(buy_signals))
    print(len(sell_signals))
    
    dataset['buy_signals'] = buy_signals
    dataset['sell_signals'] = sell_signals

    print("start 5")

    print(symblo , dataset['buy_signals'].iloc[-1] , dataset['sell_signals'].iloc[-1])

    plt.figure(figsize=(19,6))
    plt.plot(dataset['close'] , label= "Share Price" , color = "lightgray")
    plt.plot(dataset[f'SMA_{ma_1}'] , label= f"SMA_{ma_1}" , color = "orange", linestyle="--")
    plt.plot(dataset[f'SMA_{ma_2}'] , label= f"SMA_{ma_2}" , color = "pink" , linestyle="--")
    plt.scatter(dataset.index , dataset['buy_signals'] , label= "buy_signals" , marker="^" , color = "#00ff00" , lw = 3)
    plt.scatter(dataset.index , dataset['sell_signals'] , label= "sell_signals" , marker="v" , color = "#ff0000" , lw = 3)
    plt.legend(loc="upper left")
    plt.show()





def ProcessMFI( symblo):
    filename = symblo+".csv"
    dataset = dataset = pd.read_csv(filename)
    os.remove(filename)
    #cailulate the typical price
    typical_price = (dataset['close'] + dataset['high'] + dataset['low']) / 3
    print(typical_price)
    # khoang thoi gian lay du lieu
    period = 14
    # tinh money flow
    money_flow = typical_price * dataset['volume']
    # get all of the positive 
    positive_flow = []
    negative_flow = []

    for i in range(1 , len(typical_price)) :
        if typical_price[i] > typical_price[i-1]:
            positive_flow.append(money_flow[i-1])
            negative_flow.append(0)
        elif typical_price[i] < typical_price[i-1]:
            negative_flow.append(money_flow[i-1])
            positive_flow.append(0)
        else:
            positive_flow.append(0)
            negative_flow.append(0)
    
    # get all positive_flow negative_flow with time period
    positive_mf = []
    negative_mf = []

    for i in range(period - 1 , len(positive_flow)):
        positive_mf.append(sum(positive_flow[i + 1- period : i+1]))
    for i in range(period - 1 , len(negative_flow)):
        negative_mf.append(sum(negative_flow[i + 1- period : i+1]))


    # tinh money flow index
    mfi = 100 * (np.array(positive_mf) / (np.array(positive_mf) + np.array(negative_mf) ))

    print(mfi)

    dataset2 = pd.DataFrame()
    dataset2['MFI'] = mfi

    new_dataset = pd.DataFrame()
    new_dataset = dataset[period:]
    new_dataset['MFI'] = mfi

    print(new_dataset)

     # get buy and sell 
    buy_signal = []
    sell_signal = []
    for i in range(len(new_dataset['MFI'])) :
        if new_dataset['MFI'].iloc[i] > high_mfi:
            buy_signal.append(np.nan)    
            sell_signal.append(new_dataset['close'].iloc[i])
        elif new_dataset['MFI'].iloc[i] < low_mfi:
            sell_signal.append(np.nan)    
            buy_signal.append(new_dataset['close'].iloc[i])
        else :
            buy_signal.append(np.nan)  
            sell_signal.append(np.nan)



    new_dataset['Buy'] = buy_signal
    new_dataset['Sell'] = sell_signal

    print(symblo , new_dataset['Buy'].iloc[-1] , new_dataset['Sell'].iloc[-1])
    buypoint = new_dataset['Buy'].iloc[-1]
    if buypoint > 0 :
        pusgMeassageTotele(symblo+" | buy mfi | "+str(buypoint))
    
    sellpoint = new_dataset['Sell'].iloc[-1]
    if sellpoint > 0 :
        pusgMeassageTotele(symblo+" | sell mfi| "+str(sellpoint))



    # plt.figure(figsize=(15,6))
    # plt.plot(new_dataset['close'] , label= "Close price" , alpha = 0.5 , color = "lightgray")
    # plt.scatter(new_dataset.index , new_dataset['Buy'] , label= "buy_signals" , marker="^" , color = "#00ff00" , lw = 3)
    # plt.scatter(new_dataset.index , new_dataset['Sell'] , label= "sell_signals" , marker="v" , color = "#ff0000" , lw = 3)
    # plt.legend(loc="upper left")
    # plt.show()



def processMACD( symblo):
    filename = symblo+".csv"
    dataset = dataset = pd.read_csv(filename)
    os.remove(filename)

    ShortEMA = dataset.close.ewm(span=12 , adjust=False).mean()
    LongEMA = dataset.close.ewm(span=26 , adjust=False).mean()

    MACD = ShortEMA - LongEMA
    signal = MACD.ewm(span = 9 , adjust = False).mean()

    dataset['MACD'] = MACD
    dataset['signal'] = signal


    Buy = []
    Sell = []
    flag = -1 
    
    for i in range(0 , len(signal)) :
        if dataset['MACD'].iloc[i] > dataset['signal'].iloc[i]:
            Sell.append(np.nan)
            if flag != 1:
                Buy.append(dataset['close'].iloc[i])
                flag = 1
            else:
                Buy.append(np.nan)
        elif dataset['MACD'].iloc[i] < dataset['signal'].iloc[i]: 
            Buy.append(np.nan)
            if flag != 0:
                Sell.append(dataset['close'].iloc[i])
                flag = 0
            else:
                Sell.append(np.nan)
        else :
            Buy.append(np.nan)
            Sell.append(np.nan)

    dataset['Buy'] = Buy
    dataset['Sell'] = Sell

    print(dataset)

    plt.figure(figsize=(15,6))
    plt.plot(dataset['close'] , label= "Close price" , alpha = 0.5 , color = "lightgray")
    plt.scatter(dataset.index , dataset['Buy'] , label= "buy_signals" , marker="^" , color = "#00ff00" , lw = 3)
    plt.scatter(dataset.index , dataset['Sell'] , label= "sell_signals" , marker="v" , color = "#ff0000" , lw = 3)
    plt.legend(loc="upper left")
    plt.show()


    

    
  






# plt.figure(figsize=(19,6))
# plt.plot(dataset['close'] , label= "Share Price" , color = "lightgray")
# plt.plot(dataset[f'SMA_{ma_1}'] , label= f"SMA_{ma_1}" , color = "orange", linestyle="--")
# plt.plot(dataset[f'SMA_{ma_2}'] , label= f"SMA_{ma_2}" , color = "pink" , linestyle="--")
# plt.scatter(dataset.index , dataset['buy_signals'] , label= "buy_signals" , marker="^" , color = "#00ff00" , lw = 3)
# plt.scatter(dataset.index , dataset['sell_signals'] , label= "sell_signals" , marker="v" , color = "#ff0000" , lw = 3)
# plt.legend(loc="upper left")
# plt.show()

def pusgMeassageTotele(message):
    print("send message to telegram "+message)
    url = "https://api.telegram.org/bot" + token_telegram + "/sendMessage"
    data = {
        "chat_id": chatId_telegram,
        "text": message
    }

    response = requests.request(
        "GET",
        url,
        params=data
    )

startProcess()
schedule.every(1).hours.do(startProcess)


# schedule.every(1).minute.do(testschedule)

while True:
    schedule.run_pending()
    time.sleep(1)

# if __name__ == '__main__':
#     # schedule.every(1).minute.do(testschedule)
#     schedule.every(1).minutes.do(testschedule)

#     while True:
#         schedule.run_pending()
#         time.sleep(1)
