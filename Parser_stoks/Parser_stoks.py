from finam.export import Exporter, Market, LookupComparator
from openpyxl import Workbook
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import logging
import json
import csv
import ta




SYMBOLS =['FXRB',
          'FXMM'
          'FXRU',
          'FXRB',
          'FXWO',
          'FXWR',
          'SU26214RMFS5',
          'RU000A100089',
          'RU000A0ZZH84',
          'RU000A0ZYBS1'
        ]

start = datetime.datetime(2000, 1, 1);
end = datetime.datetime(datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day);
file_name = 'stocks' + '_' + '.json'





def main():
    exporter = Exporter()
    print('*** Current Russian ruble exchange rates ***')
    data = exporter.lookup(name=SYMBOLS[0], market=Market.ETF_MOEX)
    print(data.head())
    stock = exporter.download(data.index[0], market=Market.ETF_MOEX)
    print(stock.head())

    open_value = stock.get('<OPEN>')
    close_value = stock.get('<CLOSE>')
    high_value = stock.get('<HIGH>')
    low_value = stock.get('<LOW>')
    volume_value = stock.get('<VOL>')

    open_value.plot()
    # close_value.plot()
    # high_value.plot()
    # low_value.plot()
    # volume_value.plot()
    plt.show()

    # Clean NaN values
    stock = ta.utils.dropna(stock)

    plt.plot(stock[40500:41000].Close)
    plt.plot(stock[40700:41000].volatility_bbh, label='High BB')
    plt.plot(stock[40700:41000].volatility_bbl, label='Low BB')
    plt.plot(stock[40700:41000].volatility_bbm, label='EMA BB')
    plt.title('Bollinger Bands')
    plt.legend()
    plt.show()


    # print('*** Current Brent Oil price ***')
    # oil = exporter.lookup(name='Brent', market=Market.COMMODITIES,
    #                       name_comparator=LookupComparator.EQUALS)
    # assert len(oil) == 1
    # data = exporter.download(oil.index[0], market=Market.COMMODITIES)
    # print(data.tail(1))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main()