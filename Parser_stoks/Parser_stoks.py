from finam.export import Exporter, Market, LookupComparator
from openpyxl import Workbook
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import json
import csv

MY_SYMBOLS = ['FXRB ETF',
           'FXMM ETF',
           'FXRU ETF',
           'FXRB ETF',
           'FXWO ETF',
           'FXRW ETF'
           ]

SYMBOLS = [
    'FXCN ETF',
    'FXDE ETF',
    'FXGD ETF',
    'FXKZ ETF',
    'FXMM ETF',
    'FXRB ETF',
    'FXRL ETF',
    'FXRU ETF',
    'FXRW ETF',
    'FXTB ETF',
    'FXUS ETF',
    'FXWO ETF',
    'RUSB ETF',
    'RUSE ETF',
    'SBCB ETF',
    'SBGB ETF',
    'SBMX ETF',
    'SBRB ETF',
    'SBSP ETF',
    'TRUR ETF',
    'VTBA ETF',
    'VTBB ETF',
    'VTBE ETF',
    'VTBH ETF',
    'VTBM ETF'
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
    # list_currency = ['USDRUB_TOD', 'EURRUB_TOD', 'EURUSD_TOD', 'CNYRUB_TOD']
    # for currency in list_currency:
    #     time.sleep(2)  # sec
    #     print('\n__________________ ' + currency + ' __________________\n')
    #     ticker = exporter.lookup(name=currency, market=Market.CURRENCIES,
    #                              name_comparator=LookupComparator.EQUALS)
    #     data = exporter.download(ticker.index[0], market=Market.CURRENCIES, start_date=curr_moment)
    #     print(data.tail(1))

    # print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~ Indexes ~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    #
    # list_indexes = [
    #     'BSE Sensex (Индия)',
    #     'Bovespa (Бразилия)',
    #     'CAC 40',
    #     'CSI200 (Китай)',
    #     'CSI300 (Китай)',
    #     'D&J-Ind*',
    #     'Futsee-100*',
    #     'Hang Seng (Гонконг)',
    #     'KOSPI (Корея)',
    #     'N225Jap*',
    #     'NASDAQ 100**',
    #     'NASDAQ**',
    #     'SandP-500*',
    #     'Shanghai Composite(Китай)',
    #     'TA-125 Index',
    #     'TA-35 Index',
    #     'Индекс МосБиржи',
    #     'Индекс МосБиржи 10',
    #     'Индекс МосБиржи голубых фишек',
    #     'Индекс МосБиржи инноваций',
    #     'Индекс МосБиржи широкого рынка',
    #     'Индекс РТС',
    #     'Индекс РТС металлов и добычи',
    #     'Индекс РТС нефти и газа',
    #     'Индекс РТС потреб. сектора',
    #     'Индекс РТС телекоммуникаций',
    #     'Индекс РТС транспорта',
    #     'Индекс РТС финансов',
    #     'Индекс РТС химии и нефтехимии',
    #     'Индекс РТС широкого рынка',
    #     'Индекс РТС электроэнергетики',
    #     'Индекс гос обл RGBI',
    #     'Индекс гос обл RGBI TR',
    #     'Индекс корп обл MOEX CBICP',
    #     'Индекс корп обл MOEX CBITR',
    #     'Индекс корп обл MOEX CP 3',
    #     'Индекс корп обл MOEX CP 5',
    #     'Индекс корп обл MOEX TR 3',
    #     'Индекс корп обл MOEX TR 5',
    #     'Индекс металлов и добычи',
    #     'Индекс мун обл MOEX MBICP',
    #     'Индекс мун обл MOEX MBITR',
    #     'Индекс нефти и газа',
    #     'Индекс потребит сектора',
    #     'Индекс телекоммуникаций',
    #     'Индекс транспорта',
    #     'Индекс финансов',
    #     'Индекс химии и нефтехимии',
    #     'Индекс электроэнергетики'
    # ]
    # for index in list_indexes:
    #     time.sleep(2)  # sec
    #     print('\n__________________ ' + index + ' __________________\n')
    #     ticker = exporter.lookup(name=index, market=Market.INDEXES,
    #                              name_comparator=LookupComparator.EQUALS)
    #     data = exporter.download(ticker.index[0], market=Market.INDEXES, start_date=curr_moment)
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
