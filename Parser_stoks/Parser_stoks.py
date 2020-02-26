from finam.export import Exporter, Market, LookupComparator
from openpyxl import Workbook
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import json
import csv

SYMBOLS = ['FXRB',
           'FXMM',
           'FXRU',
           'FXRB',
           'FXWO',
           'FXWR'
           ]

curr_moment = datetime.date(datetime.datetime.now().year,
                             datetime.datetime.now().month,
                             datetime.datetime.now().day)
file_name = 'stocks' + '';
extansion = '.json'


def main():
    exporter = Exporter()
    # print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~ Goods ~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    #
    # list_goods = ['Brent', 'Бензин',
    #               'Золото', 'Серебро', 'Платина', 'Палладий',
    #               'Медь', 'Цинк', 'Алюминий',
    #               'Сахар', 'Пшеница']
    # for goods in list_goods:
    #     time.sleep(2)  # sec
    #     print('\n__________________ ' + goods + ' __________________\n')
    #     ticker = exporter.lookup(name=goods, market=Market.COMMODITIES,
    #                              name_comparator=LookupComparator.EQUALS)
    #     data = exporter.download(ticker.index[0], market=Market.COMMODITIES, start_date=curr_moment)
    #     print(data.tail(1))

    # print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~ Currency ~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    #
    # list_currency = ['USDRUB_TOD', 'EURRUB_TOD', 'EURUSD_TOD']
    # for currency in list_currency:
    #     time.sleep(2)  # sec
    #     print('\n__________________ ' + currency + ' __________________\n')
    #     ticker = exporter.lookup(name=currency, market=Market.CURRENCIES,
    #                              name_comparator=LookupComparator.EQUALS)
    #     data = exporter.download(ticker.index[0], market=Market.CURRENCIES, start_date=curr_moment)
    #     print(data.tail(1))

    print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~ Stock ~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

    list_stocks = SYMBOLS
    for stock in list_stocks:
        time.sleep(2)  # sec
        print('\n__________________ ' + stock + ' __________________\n')
        ticker = exporter.lookup(name=stock, market=Market.ETF_MOEX,
                                 name_comparator=LookupComparator.EQUALS)
        data = exporter.download(ticker.index[0], market=Market.ETF_MOEX, start_date=curr_moment)
        print(data.tail(1))


    # data = exporter.lookup(name=SYMBOLS[0], market=Market.ETF_MOEX)
    # # print(data.head())
    # stock = exporter.download(data.index[0], market=Market.ETF_MOEX, start_date=start)
    # # print(stock.head())
    #
    # open_value = stock.get('<OPEN>')
    # close_value = stock.get('<CLOSE>')
    # high_value = stock.get('<HIGH>')
    # low_value = stock.get('<LOW>')
    # volume_value = stock.get('<VOL>')
    #
    # # open_value.plot()
    # # close_value.plot()
    # # high_value.plot()
    # # low_value.plot()
    # # volume_value.plot()
    # # plt.show()
    #
    # stock.to_csv(file_name + '.csv')
    #
    # # Load datas
    # df = pd.read_csv(file_name + '.csv', sep=',')

if __name__ == '__main__':
    main()
