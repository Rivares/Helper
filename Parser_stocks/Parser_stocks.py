from finam.export import Exporter, Market, LookupComparator
from openpyxl import Workbook
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import json
import csv

root_path = 'C:\\Users\\user\\0_Py\\'

MY_SYMBOLS = ['FXRB ETF',
               'FXMM ETF',
               'FXRU ETF',
               'FXRB ETF',
               'FXWO ETF',
               'FXRW ETF'
               ]

curr_moment = datetime.date(datetime.datetime.now().year,
                            datetime.datetime.now().month,
                            datetime.datetime.now().day)


def write_data_json(data, path, file_name):
    extension = '.json'

    with open(path + file_name + extension, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def main():
    exporter = Exporter()

    market = []
    list_goods = []
    list_currency = []
    list_indexes = []
    list_stocks = []

    # print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~ Goods ~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

    list_name_goods = [
                       'Brent',
                       'Natural Gas',
                       'Алюминий',
                       'Бензин',
                       'Золото',
                       'Мазут',
                       'Медь',
                       'Никель',
                       'Палладий',
                       'Платина',
                       'Пшеница',
                       'Серебро'
                      ]

    for goods in list_name_goods:
        time.sleep(2)  # sec
        # print('\n__________________ ' + goods + ' __________________\n')
        ticker = exporter.lookup(name=goods, market=Market.COMMODITIES,
                                 name_comparator=LookupComparator.EQUALS)
        data = exporter.download(ticker.index[0], market=Market.COMMODITIES, start_date=curr_moment)

        open_value = data.get('<OPEN>')
        close_value = data.get('<CLOSE>')
        high_value = data.get('<HIGH>')
        low_value = data.get('<LOW>')
        volume_value = data.get('<VOL>')

        list_open_value = open_value.to_list()
        list_close_value = close_value.to_list()
        list_high_value = high_value.to_list()
        list_low_value = low_value.to_list()
        list_volume_value = volume_value.to_list()

        list_goods.append({"open_value": list_open_value[-1],
                           "close_value": list_close_value[-1],
                           "high_value": list_high_value[-1],
                           "low_value": list_low_value[-1],
                           "volume_value": list_volume_value[-1]})

        # print(data.tail(1))

    # print(list_goods)
    # print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~ Currency ~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

    list_name_currency = [
                          'USDRUB_TOD',
                          'EURRUB_TOD',
                          'EURUSD_TOD',
                          'CNYRUB_TOD'
                         ]

    for currency in list_name_currency:
        time.sleep(2)  # sec
        # print('\n__________________ ' + currency + ' __________________\n')
        ticker = exporter.lookup(name=currency, market=Market.CURRENCIES,
                                 name_comparator=LookupComparator.EQUALS)
        data = exporter.download(ticker.index[0], market=Market.CURRENCIES, start_date=curr_moment)

        open_value = data.get('<OPEN>')
        close_value = data.get('<CLOSE>')
        high_value = data.get('<HIGH>')
        low_value = data.get('<LOW>')
        volume_value = data.get('<VOL>')

        list_open_value = open_value.to_list()
        list_close_value = close_value.to_list()
        list_high_value = high_value.to_list()
        list_low_value = low_value.to_list()
        list_volume_value = volume_value.to_list()

        list_currency.append({"open_value": list_open_value[-1],
                              "close_value": list_close_value[-1],
                              "high_value": list_high_value[-1],
                              "low_value": list_low_value[-1],
                              "volume_value": list_volume_value[-1]})
        # print(data.tail(1))

    # print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~ Indexes ~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

    list_name_indexes = [
        'BSE Sensex (Индия)',
        'Bovespa (Бразилия)',
        'CAC 40',
        'CSI200 (Китай)',
        'CSI300 (Китай)',
        'D&J-Ind*',
        'Futsee-100*',
        'Hang Seng (Гонконг)',
        'KOSPI (Корея)',
        'N225Jap*',
        'NASDAQ 100**',
        'NASDAQ**',
        'SandP-500*',
        'Shanghai Composite(Китай)',
        'TA-125 Index',
        'TA-35 Index',
        'Индекс МосБиржи',
        'Индекс МосБиржи 10',
        'Индекс МосБиржи голубых фишек',
        'Индекс МосБиржи инноваций',
        'Индекс МосБиржи широкого рынка',
        'Индекс РТС',
        'Индекс РТС металлов и добычи',
        'Индекс РТС нефти и газа',
        'Индекс РТС потреб. сектора',
        'Индекс РТС телекоммуникаций',
        'Индекс РТС транспорта',
        'Индекс РТС финансов',
        'Индекс РТС химии и нефтехимии',
        'Индекс РТС широкого рынка',
        'Индекс РТС электроэнергетики',
        'Индекс гос обл RGBI',
        'Индекс гос обл RGBI TR',
        'Индекс корп обл MOEX CBICP',
        'Индекс корп обл MOEX CBITR',
        'Индекс корп обл MOEX CP 3',
        'Индекс корп обл MOEX CP 5',
        'Индекс корп обл MOEX TR 3',
        'Индекс корп обл MOEX TR 5',
        'Индекс металлов и добычи',
        'Индекс мун обл MOEX MBICP',
        'Индекс мун обл MOEX MBITR',
        'Индекс нефти и газа',
        'Индекс потребит сектора',
        'Индекс телекоммуникаций',
        'Индекс транспорта',
        'Индекс финансов',
        'Индекс химии и нефтехимии',
        'Индекс электроэнергетики'
    ]

    for index in list_name_indexes:
        time.sleep(2)  # sec
        # print('\n__________________ ' + index + ' __________________\n')
        ticker = exporter.lookup(name=index, market=Market.INDEXES,
                                 name_comparator=LookupComparator.EQUALS)
        data = exporter.download(ticker.index[0], market=Market.INDEXES, start_date=curr_moment)

        open_value = data.get('<OPEN>')
        close_value = data.get('<CLOSE>')
        high_value = data.get('<HIGH>')
        low_value = data.get('<LOW>')
        volume_value = data.get('<VOL>')

        list_open_value = open_value.to_list()
        list_close_value = close_value.to_list()
        list_high_value = high_value.to_list()
        list_low_value = low_value.to_list()
        list_volume_value = volume_value.to_list()

        list_indexes.append({"open_value": list_open_value[-1],
                             "close_value": list_close_value[-1],
                             "high_value": list_high_value[-1],
                             "low_value": list_low_value[-1],
                             "volume_value": list_volume_value[-1]})

    # print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~ Stock ~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

    list_name_stocks = [
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

    for stock in list_name_stocks:
        time.sleep(2)  # sec
        # print('\n__________________ ' + stock + ' __________________\n')
        ticker = exporter.lookup(name=stock, market=Market.ETF_MOEX,
                                 name_comparator=LookupComparator.EQUALS)
        data = exporter.download(ticker.index[0], market=Market.ETF_MOEX, start_date=curr_moment)

        open_value = data.get('<OPEN>')
        close_value = data.get('<CLOSE>')
        high_value = data.get('<HIGH>')
        low_value = data.get('<LOW>')
        volume_value = data.get('<VOL>')

        list_open_value = open_value.to_list()
        list_close_value = close_value.to_list()
        list_high_value = high_value.to_list()
        list_low_value = low_value.to_list()
        list_volume_value = volume_value.to_list()

        list_stocks.append({"open_value": list_open_value[-1],
                            "close_value": list_close_value[-1],
                            "high_value": list_high_value[-1],
                            "low_value": list_low_value[-1],
                            "volume_value": list_volume_value[-1]})

    market.append(list_goods)
    market.append(list_currency)
    market.append(list_indexes)
    market.append(list_stocks)

    path = root_path + 'Helper\\Parser_stocks\\'
    file_name_market = 'market'

    write_data_json(market, path, file_name_market)


if __name__ == '__main__':
    main()
