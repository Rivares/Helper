from finam.export import Exporter, Market, LookupComparator
from openpyxl import Workbook
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import logging
import json
import csv
import ta

SYMBOLS = ['FXRB',
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
file_name = 'stocks' + '_';
extansion = '.json'


def main():
    exporter = Exporter()
    data = exporter.lookup(name=SYMBOLS[0], market=Market.ETF_MOEX)
    # print(data.head())
    stock = exporter.download(data.index[0], market=Market.ETF_MOEX)
    # print(stock.head())

    open_value = stock.get('<OPEN>')
    close_value = stock.get('<CLOSE>')
    high_value = stock.get('<HIGH>')
    low_value = stock.get('<LOW>')
    volume_value = stock.get('<VOL>')

    # open_value.plot()
    # close_value.plot()
    # high_value.plot()
    # low_value.plot()
    # volume_value.plot()
    # plt.show()

    stock.to_csv(file_name + '.csv')

    # Load datas
    df = pd.read_csv(file_name + '.csv', sep=',')

    # Clean NaN values
    df = ta.utils.dropna(df)

    # # _____________________________________________________________________________________________________
    # # _______________________________________ Volatility Inidicators ______________________________________
    # # _____________________________________________________________________________________________________
    # # __________________________________________ Bollinger Bands __________________________________________
    #
    # # Initialize Bollinger Bands Indicator
    #
    # indicator_bb = ta.volatility.BollingerBands(close=df["<CLOSE>"], n=20, ndev=2, fillna=True)
    #
    # # Add Bollinger Bands features
    # df['bb_bbm'] = indicator_bb.bollinger_mavg()
    # df['bb_bbh'] = indicator_bb.bollinger_hband()
    # df['bb_bbl'] = indicator_bb.bollinger_lband()
    #
    # # Add Bollinger Band high indicator
    # df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()
    #
    # # Add Bollinger Band low indicator
    # df['bb_bbli'] = indicator_bb.bollinger_lband_indicator()
    #
    # # Add width size Bollinger Bands
    # df['bb_bbw'] = indicator_bb.bollinger_wband()
    #
    # print(df.columns)
    #
    # plt.plot(df["<CLOSE>"])
    # plt.plot(df['bb_bbh'], label='High BB')
    # plt.plot(df['bb_bbl'], label='Low BB')
    # plt.plot(df['bb_bbm'], label='EMA BB')
    # plt.title('Bollinger Bands')
    # plt.legend()
    # plt.show()
    #
    # # __________________________________________ Keltner Channel __________________________________________
    #
    # # Initialize Keltner Channel Indicator
    # indicator_kc = ta.volatility.KeltnerChannel(high=df["<HIGH>"], low=df["<LOW>"], close=df["<CLOSE>"], n=20,
    #                                             fillna=True)
    #
    # # Add Keltner Channel features
    # df['kc_kcc'] = indicator_kc.keltner_channel_central()
    # df['kc_kch'] = indicator_kc.keltner_channel_hband()
    # df['kc_kcl'] = indicator_kc.keltner_channel_lband()
    #
    # # Add Keltner Channel high indicator
    # df['kc_bbhi'] = indicator_kc.keltner_channel_hband_indicator()
    #
    # # Add Keltner Channel low indicator
    # df['kc_bbli'] = indicator_kc.keltner_channel_lband_indicator()
    #
    # plt.plot(df["<CLOSE>"])
    # plt.plot(df['kc_kcc'], label='Central KC')
    # plt.plot(df['kc_kch'], label='High KC')
    # plt.plot(df['kc_kcl'], label='Low KC')
    # plt.title('Keltner Channel')
    # plt.legend()
    # plt.show()
    #
    # # __________________________________________ Average true range (ATR) __________________________________________
    #
    # # Initialize Average true range Indicator
    # indicator_atr = ta.volatility.AverageTrueRange(high=df["<HIGH>"], low=df["<LOW>"], close=df["<CLOSE>"], n=20, fillna=True)
    #
    # # Add ATR indicator
    # df['atr_i'] = indicator_atr.average_true_range()
    #
    # plt.plot(df["<CLOSE>"])
    # plt.plot(df['atr_i'], label='ATR')
    # plt.title('Average true range (ATR)')
    # plt.legend()
    # plt.show()
    #
    # # __________________________________________ Donchian Channel __________________________________________
    #
    # # Initialize Donchian Channel Indicator
    # indicator_dc = ta.volatility.DonchianChannel(close=df["<CLOSE>"], n=20, fillna=True)
    #
    # # Add Donchian Channel features
    # df['dc_dch'] = indicator_dc.donchian_channel_hband()
    # df['dc_dcl'] = indicator_dc.donchian_channel_lband()
    #
    # # Add Donchian Channel high indicator
    # df['dc_dchi'] = indicator_dc.donchian_channel_hband_indicator()
    #
    # # Add Donchian Channel low indicator
    # df['dc_dcli'] = indicator_dc.donchian_channel_lband_indicator()
    #
    # plt.plot(df["<CLOSE>"])
    # plt.plot(df['dc_dch'], label='High DC')
    # plt.plot(df['dc_dcl'], label='Low DC')
    # plt.title('Donchian Channel')
    # plt.legend()
    # plt.show()
    #
    # # _____________________________________________________________________________________________________
    # # __________________________________________ Trend Indicators _________________________________________
    # # _____________________________________________________________________________________________________
    # _____________________________ Average Directional Movement Index (ADX) ________________________________

    # Initialize ADX Indicator
    indicator_adx = ta.trend.ADXIndicator(high=df["<HIGH>"], low=df["<LOW>"], close=df["<CLOSE>"], n=20, fillna=True)

    # Add ADX features
    df['adx_aver'] = indicator_adx.adx()
    df['adx_DI_pos'] = indicator_adx.adx_pos()
    df['adx_DI_neg'] = indicator_adx.adx_neg()

    plt.plot(df["<CLOSE>"])
    plt.plot(df['adx_aver'], label='ADX')
    plt.plot(df['adx_DI_pos'], label='+DI')
    plt.plot(df['adx_DI_neg'], label='-DI')
    plt.title('ADX')
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
