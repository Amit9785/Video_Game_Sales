from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from functools import reduce
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

def posix_time(dt):
    return (dt - datetime(1970, 1, 1)) / timedelta(seconds=1)

data = pd.read_csv('Train.csv')
data = data.sort_values(by=['date_time'], ascending=True).reset_index(drop=True)
last_n_hours = [1, 2, 3, 4, 5, 6]

for n in last_n_hours:
    data[f'last_{n}_hour_traffic'] = data['traffic_volume'].shift(n)
data = data.dropna().reset_index(drop=True)
data.loc[data['is_holiday'] != 'None', 'is_holiday'] = 1
data.loc[data['is_holiday'] == 'None', 'is_holiday'] = 0
data['is_holiday'] = data['is_holiday'].astype(int)

data['date_time'] = pd.to_datetime(data['date_time'])
data['hour'] = data['date_time'].dt.hour
data['month_day'] = data['date_time'].dt.day
data['weekday'] = data['date_time'].dt.weekday + 1
data['month'] = data['date_time'].dt.month
data['year'] = data['date_time'].dt.year

data.to_csv("traffic_volume_data.csv", index=None)

sns.set()
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_csv("traffic_volume_data.csv")
data = data.sample(min(10000, len(data))).reset_index(drop=True)

label_columns = ['weather_type', 'weather_description']
numeric_columns = ['is_holiday', 'temperature', 'weekday', 'hour', 'month_day', 'year', 'month']
features = numeric_columns + label_columns
X = data[features]

def unique(list1):
    ans = reduce(lambda re, x: re+[x] if x not in re else re, list1, [])
    print(ans)

n1 = data['weather_type']
n2 = data['weather_description']
unique(n1)
unique(n2)
n1features = ['Rain', 'Clouds', 'Clear', 'Snow', 'Mist', 'Drizzle', 'Haze', 'Thunderstorm', 'Fog', 'Smoke', 'Squall']
n2features = ['light rain', 'few clouds', 'Sky is Clear', 'light snow', 'sky is clear', 'mist', 'broken clouds', 'moderate rain', 'drizzle', 'overcast clouds', 'scattered clouds', 'haze', 'proximity thunderstorm', 'light intensity drizzle', 'heavy snow', 'heavy intensity rain', 'fog', 'heavy intensity drizzle', 'shower snow', 'snow', 'thunderstorm with rain', 'thunderstorm with heavy rain', 'thunderstorm with light rain', 'proximity thunderstorm with rain', 'thunderstorm with drizzle', 'smoke', 'thunderstorm', 'proximity shower rain', 'very heavy rain', 'proximity thunderstorm with drizzle', 'light rain and snow', 'light intensity shower rain', 'SQUALLS', 'shower drizzle', 'thunderstorm with light drizzle']

n11 = []
n22 = []
for i in range(len(data)):
    n11.append(n1features.index(n1[i]) + 1 if n1[i] in n1features else 0)
    n22.append(n2features.index(n2[i]) + 1 if n2[i] in n2features else 0)

data['weather_type'] = n11
data['weather_description'] = n22

features = numeric_columns + label_columns
target = ['traffic_volume']
X = data[features]
y = data[target]

x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X)

y_scaler = MinMaxScaler()
y = y_scaler.fit_transform(y).flatten()

metrics = ['month', 'month_day', 'weekday', 'hour']
fig = plt.figure(figsize=(8, 4*len(metrics)))
for i, metric in enumerate(metrics):
    ax = fig.add_subplot(len(metrics), 1, i+1)
    ax.plot(data.groupby(metric)['traffic_volume'].mean(), '-o')
    ax.set_xlabel(metric)
    ax.set_ylabel("Mean Traffic")
    ax.set_title(f"Traffic Trend by {metric}")

plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2)

# Train ARIMA model
model = ARIMA(trainY, order=(5,1,0))
model_fit = model.fit()

# Make predictions
y_pred = model_fit.forecast(steps=len(testX))

# Calculate R-squared
r_squared = r2_score(testY, y_pred)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(testY, y_pred)

# Calculate Root Mean Square Error (RMSE)
rmse = np.sqrt(mean_squared_error(testY, y_pred))

# Define a function to calculate Mean Absolute Percent Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Calculate MAPE
mape = mean_absolute_percentage_error(testY, y_pred)

print('R-squared:', r_squared)
print('Mean Absolute Error:', mae)
print('Root Mean Square Error:', rmse)
print('Mean Absolute Percent Error:', mape)

# Sample input
ip = [0, 89, 2, 288.28, 1, 9, 2, 2012, 10]
ip = x_scaler.transform([ip])
out = model_fit.forecast(steps=len(ip))
print('Before inverse Scaling :', out)
y_pred = y_scaler.inverse_transform([out])
print('Traffic Volume : ', y_pred)

# Interpretation of Traffic Volume
if y_pred <= 1000:
    print("No Traffic ")
elif 1000 < y_pred <= 3000:
    print("Busy or Normal Traffic")
elif 3000 < y_pred <= 5500:
    print("Heavy Traffic")
else:
    print("Worst case")
