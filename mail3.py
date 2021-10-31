# number of total past observations from the original dataset to be considered
n_past_total = 1200

# number of past observations to be considered for the LSTM training and prediction
n_past = 100

# number of future datapoints to predict (if higher than 1, the model switch to Multi-Step)
n_future = 10

# activation function used for the RNN (softsign, relu, sigmoid)
activation = 'softsign'

# dropout for the hidden layers
dropout = 0.2

# number of hidden layers
n_layers = 8

# number of neurons of the hidden layers
n_neurons = 20

# features to be considered for training (if only one is Close, then its Univariate, if more, then it's Multivariate)
features = ['close', 'volume']
#features = ['Close']

# number of inputs features (if higher than 1, )
n_features = len(features)

# patience for the early stopping (number of epochs)
patience = 25

# optimizer (adam, RMSprop)
optimizer='adam'

import os
import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x) #avoid scientific notation
import datetime 
import math
from matplotlib import pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import Image
import pandas_datareader as web
from binance.client import Client
import json


# YOUR API KEYS HERE
api_key = "lmRl0bcG4my9tvoNDdgUCEuFKASmzwEPSUnvl4GIm3OUdNpdloQtGiTZcYioDen8"    #Enter your own API-key here
api_secret = "DgRJaDCnJvx5FJbx1qXbiduBbcLqtXxxjtxl360gC7RJnErlzQ5FnIC6hWa5Ydh3" #Enter your own API-secret here

# get data
crypto_currency = 'BTC'
against_currency = 'USD'


def ProcessTool(dataset1 , time) :
    dataset = pd.read_csv("data.csv")
    print(dataset)

    # checking if close is not equal to adj close
    dataset[dataset['close']!=dataset['close']]
    dataset.describe()

    # use close only and fill NaN with ffil
    df = dataset.set_index('timestamp')[features]
    # #.tail(n_past_total)
    df = df.set_index(pd.to_datetime(df.index))
    df.fillna(method='ffill',inplace=True)

    # # looking at the correlation of the main possible variables
    dataset[['close','volume']].corr()

    print(df)

    # plotting Closing Price and volume
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(go.Scatter(x=dataset['close'].index, y=dataset['close'].values, name='close'), secondary_y=False)

    fig.add_trace(go.Scatter(x=dataset['volume'].index, y=dataset['volume'].values, name='volume'), secondary_y=True)

    # Add figure title
    fig.update_layout(title_text="BTC: {}, {}".format('close', 'volume'))

    # Set x-axis title
    fig.update_xaxes(title_text='<b>timestamp</b>')

    # Set y-axes titles
    fig.update_yaxes(title_text='<b>close</b>', secondary_y=False)
    fig.update_yaxes(title_text='<b>volume</b>', secondary_y=True)

    # droppping firsts observations becuase they may not be representative of BTC behaviour now due to beginnings of crypto market
    df = df.tail(n_past_total)

    # train test split
    training_set = df.values
    print('training_set.shape:\t', training_set.shape)

    # scale
    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(training_set)
    print('training_set_scaled.shape: ', training_set_scaled.shape)
    training_set_scaled

    # creating a data structure with 60 timesteps and 1 output
    X_train = []
    y_train = []

    for i in range(n_past, len(training_set_scaled) - n_future + 1):
        X_train.append(training_set_scaled[i-n_past:i, :])
        y_train.append(training_set_scaled[i:i+n_future, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train.shape, y_train.shape

    # reshaping (needed to fit RNN)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_features))
    X_train.shape

    # Building the RNN

    # Initialising the RNN
    regressor = Sequential()

    # Input layer
    regressor.add(LSTM(units=n_past, return_sequences=True, activation=activation, input_shape=(X_train.shape[1], n_features))) 
    #regressor.add(LSTM(units=neurons, return_sequences=True, activation=activation, input_shape=(X_train.shape[1], 1))) 

    # Hidden layers
    for _ in range(n_layers):
        regressor.add(Dropout(dropout))
        regressor.add(LSTM(units=n_neurons, return_sequences=True, activation=activation))

    # Last hidden layer (changing the return_sequences)
    regressor.add(Dropout(dropout))
    regressor.add(LSTM(units=n_neurons, return_sequences=False, activation=activation))

    # Adding the output layer
    regressor.add(Dense(units=n_future))

    # Compiling the RNN
    regressor.compile(optimizer=optimizer, loss='mse')

    # Model summary
    regressor.summary()

    # Adding early stopping
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)

    # Fitting the RNN to the Training set
    res = regressor.fit(X_train, y_train
                        , batch_size=64
                        , epochs=750
                        , validation_split=0.1
                        , callbacks=[early_stop]
                    )


    # Exporting the regressor
    last_date = dataset.timestamp.values[-1]
    params = ['reg', n_past_total, n_past, n_future, activation, n_layers, n_neurons, n_features, patience, optimizer]
    modelname = 'output/'
    for i in params:
        print(i)
        modelname += str(i)
        if i!= params[-1]:
            modelname += '_'
    if not os.path.exists(modelname):
        print(modelname)
        os.makedirs(modelname)
    regressor.save('{}/regressor.h5'.format(modelname))

    list(res.history)

    results = res

    history = results.history
    plt.figure(figsize=(12,4))
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['val_loss', 'loss'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('{}/Loss.png'.format(modelname))
    plt.show()


    def dummy_invscaler(y, n_features):
        '''
        Since the scaler was trained into 2 features, it needs two features to perform the inverse scaleer.
        For that purpose, this function will create a dummy array and concatenate it to the y_pred/y_true.
        That dummy of ones will be drop after performing the inverse_transform.
        INPUTS: array 'y', shape (X,)
        '''
        y = np.array(y).reshape(-1,1)
        if n_features>1:
            dummy = np.ones((len(y), n_features-1))
            y = np.concatenate((y, dummy), axis=1)
            y = sc.inverse_transform(y)
            y = y[:,0]
        else:
            y = sc.inverse_transform(y)
        return y


    # Validation

    # getting the predictions
    y_pred = regressor.predict(X_train[-1].reshape(1, n_past, n_features)).tolist()[0]
    y_pred = dummy_invscaler(y_pred, n_features)

    # creating a DF of the predicted prices
    y_pred_df = pd.DataFrame(y_pred, 
                            index=df[['close']].tail(n_future).index, 
                            columns=df[['close']].columns)

    # getting the true values
    y_true_df = df[['close']].tail(n_past)
    y_true = y_true_df.tail(n_future).values

    print('y_pred:\n', y_pred.tolist())
    print('y_true:\n', y_true.tolist())

    # plotting the results
    plt.figure(figsize=(16,5))
    plt.plot(y_pred_df, label='Predicted')
    plt.plot(y_true_df, label='True')

    plt.title('BTC price Predicted vs True')
    plt.legend()
    plt.savefig('{}/Validation.png'.format(modelname))
    plt.show()

    # Root Mean Square Error (RMSE)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    rmse

    # Mean Square Error (MSE)
    mse = mean_absolute_error(y_true, y_pred)
    mse

    #Explained variance regression score function.
    explained_variance_score(y_true, y_pred)
    # Best possible score is 1.0, lower values are worse



    # Predicting/Forecasting

    # getting the predictions
    x = df[features][-n_past:].values
    x = sc.transform(x)
    y_pred = regressor.predict(x.reshape(1, n_past, n_features)).tolist()[0]
    y_pred = dummy_invscaler(y_pred, n_features)

    # creating a DF of the predicted prices
    # y_pred_df = pd.DataFrame(y_pred, 
    #                         index=pd.date_range(start=df[['close']].index[-1]+datetime.timedelta(minutes=15),
    #                                             periods=len(y_pred), 
    #                                             freq="15 min"), 
    #                         columns=df[['close']].columns)

    y_pred_df = pd.DataFrame(y_pred, 
                            index=pd.date_range(start=df[['close']].index[-1]+datetime.timedelta(hours=1),
                                                periods=len(y_pred), 
                                                freq="H"), 
                            columns=df[['close']].columns)

    # getting the true values
    y_true_df = df[['close']].tail(n_past)

    # linking them
    #y_true_df = y_true_df.append(y_pred_df.head(1))
    y_pred_df = y_pred_df.append(y_true_df.tail(1)).sort_index()


    print('y_pred:\n',y_pred_df )
    print('y_true:\n', y_true.tolist())

    # plotting the results
    plt.figure(figsize=(16,5))
    plt.plot(y_pred_df, label='Predicted')
    plt.plot(y_true_df, label='True')

    plt.title('BTC price Predicted vs True')
    plt.legend()
    plt.savefig('{}/Predictions.png'.format(modelname))
    print(modelname)
    plt.show()


def binanceBarExtractor(symbol , time):
    print("Start bot !!!!!!")
    bclient = Client(api_key=api_key, api_secret=api_secret)
    filename = "data.csv"

    start_date = datetime.datetime.strptime('1 Jun 2020', '%d %b %Y')

    today = datetime.datetime.now()
    print(today)
    
    klines = bclient.get_historical_klines(symbol, time, start_date.strftime("%d %b %Y %H:%M:%S"), today.strftime("%d %b %Y %H:00:00"), 1000)
    data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')

    data.set_index('timestamp', inplace=True)
    data.to_csv(filename)
    # ProcessTool(data, time)
    # os.remove("data.csv")
    # print('finished!')


def binacePrice(symbol):
    bclient = Client(api_key=api_key, api_secret=api_secret)
    bclient.API_URL = 'https://testnet.binance.vision/api'

    avg_price = bclient.get_avg_price(symbol=symbol)
    print(avg_price)


    balance = bclient.get_asset_balance(asset='BTC')
    print(balance)


def getallticker():
    bclient = Client(api_key=api_key, api_secret=api_secret)
    prices = bclient.get_all_tickers()
    for ticker in prices:
        if str(ticker['symbol'])[-4:] == 'USDT':
            print(ticker['symbol'])
        


if __name__ == '__main__':
        # Obviously replace BTCUSDT with whichever symbol you want from binance
    # Wherever you've saved this code is the same directory you will find the resulting CSV file
    # binanceBarExtractor('BTCUSDT', Client.KLINE_INTERVAL_15MINUTE)
    binanceBarExtractor('BTCUSDT', Client.KLINE_INTERVAL_1HOUR)
    # binacePrice('BTCUSDT')
    # getallticker()