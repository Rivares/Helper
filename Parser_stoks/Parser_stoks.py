from openpyxl import Workbook
import matplotlib.pyplot as plt
from pandas_datareader import data
import pandas as pd
import datetime
import csv
import ta



SYMBOL = 'FXRB'
# SYMBOL = 'FXMM'
# SYMBOL = 'FXRU'
# SYMBOL = 'FXRB'
# SYMBOL = 'FXWO'
# SYMBOL = 'FXWR'
# SYMBOL = 'SU26214RMFS5'
# SYMBOL = 'RU000A100089'
# SYMBOL = 'RU000A0ZZH84'
# SYMBOL = 'RU000A0ZYBS1'

start = datetime.datetime(2000, 1, 1);
end = datetime.datetime(datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day);

stock = data.DataReader(name = SYMBOL, data_source = "moex", start = start, end = end)

file_name = 'stock_' + SYMBOL + '.csv'
stock.to_csv(file_name);

wb = Workbook()
ws = wb.active
with open(file_name, 'r') as f:
    for row in csv.reader(f):
        ws.append(row)
wb.save('stock_' + SYMBOL + '.xlsx')


y1 = stock.get('OPEN')

y1.plot(title=f'{SYMBOL} Close Price')

plt.show()

# Load datas
df = pd.read_csv(file_name, sep=',')

# Clean NaN values
df = ta.utils.dropna(df)


stock['rsi'] = ta.momentum.rsi(y1)
y2 = stock.get('rsi')
print(y2)

y2.plot(title=f'{SYMBOL} Close Price')


plt.show()






