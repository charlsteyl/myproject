import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import time
import pickle

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close']*100.0
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open']*100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df))) # take any value and round p to the next interger whole. Predict out 10 % of the whole'''
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out) # need to be int because you shift column up. Adjusted close price 10 days into the future

#print(df.head())

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:] # stuff we are going to predict against
X = X[:-forecast_out]

df.dropna(inplace = True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2) # this function takes the dataset, shuffles it then outputs a percantage for training

##clf = LinearRegression(n_jobs=-1)
##clf.fit(X_train, y_train)
##with open('linearregression.pickle', 'wb') as f:
##    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan #specifies column full of not a number data

last_date = df.iloc[-1].name
print time.mktime(last_date.to_pydatetime().timetuple())

one_day = 86400
last_unix = time.mktime(last_date.to_pydatetime().timetuple()) - one_day

next_unix = last_unix + one_day # does not know the actual date

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)  #
    next_unix += one_day                                    #
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] +[i] # iterating through forecast and day set taking making future features nans and final column is i which is forecast

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
