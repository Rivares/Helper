# coding: UTF-8

import lib_general as my_general

root_path = my_general.root_path
curr_ticker = my_general.name_ticker

curr_path = root_path + 'Helper\\TA_stocks\\'

start = my_general.datetime.date(my_general.datetime.datetime.now().year - 1,
                                 my_general.datetime.datetime.now().month,
                                 my_general.datetime.datetime.now().day)

curr_moment = my_general.datetime.date(my_general.datetime.datetime.now().year,
                                       my_general.datetime.datetime.now().month,
                                       my_general.datetime.datetime.now().day)


def main():
    print("\n__________________ TA " + "(FXRB)" + " __________________\n")

    exporter = my_general.Exporter()

    target_ticker = curr_ticker

    list_indicators_target_ticker = []

    list_tiker_params = []
    data = exporter.lookup(name=target_ticker, market=my_general.Market.ETF_MOEX)
    # print(data.head())
    stock = exporter.download(data.index[0], market=my_general.Market.ETF_MOEX, start_date=start, end_date=curr_moment)
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

    stock.to_csv(curr_path + 'target_ticker' + '.csv')

    list_open_value = open_value.to_list()
    list_close_value = close_value.to_list()
    list_high_value = high_value.to_list()
    list_low_value = low_value.to_list()
    list_volume_value = volume_value.to_list()

    # Load datas
    df = my_general.pd.read_csv(curr_path + 'target_ticker' + '.csv', sep=',')

    # Clean NaN values
    df = my_general.ta.utils.dropna(df)

    # _____________________________________________________________________________________________________
    # _______________________________________ Volatility Inidicators ______________________________________
    # _____________________________________________________________________________________________________
    # __________________________________________ Bollinger Bands __________________________________________

    # Initialize Bollinger Bands Indicator
    indicator_bb = my_general.ta.volatility.BollingerBands(close=df["<CLOSE>"], n=20, ndev=2, fillna=True)

    # Add Bollinger Bands features
    df['bb_bbm'] = indicator_bb.bollinger_mavg()
    df['bb_bbh'] = indicator_bb.bollinger_hband()
    df['bb_bbl'] = indicator_bb.bollinger_lband()

    # Add Bollinger Band high indicator
    df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()

    # Add Bollinger Band low indicator
    df['bb_bbli'] = indicator_bb.bollinger_lband_indicator()

    # Add width size Bollinger Bands
    df['bb_bbw'] = indicator_bb.bollinger_wband()

    # print(df.columns)
    #
    # plt.plot(df["<CLOSE>"])
    # plt.plot(df['bb_bbh'], label='High BB')
    # plt.plot(df['bb_bbl'], label='Low BB')
    # plt.plot(df['bb_bbm'], label='EMA BB')
    # plt.title('Bollinger Bands')
    # plt.legend()
    # plt.show()

    bb_bbh = df['bb_bbh'].to_list()
    bb_bbl = df['bb_bbl'].to_list()
    bb_bbm = df['bb_bbm'].to_list()

    # __________________________________________ Keltner Channel __________________________________________

    # Initialize Keltner Channel Indicator
    indicator_kc = my_general.ta.volatility.KeltnerChannel(high=df["<HIGH>"],
                                                       low=df["<LOW>"], close=df["<CLOSE>"], n=20, fillna=True)

    # Add Keltner Channel features
    df['kc_kcc'] = indicator_kc.keltner_channel_central()
    df['kc_kch'] = indicator_kc.keltner_channel_hband()
    df['kc_kcl'] = indicator_kc.keltner_channel_lband()

    # Add Keltner Channel high indicator
    df['kc_bbhi'] = indicator_kc.keltner_channel_hband_indicator()

    # Add Keltner Channel low indicator
    df['kc_bbli'] = indicator_kc.keltner_channel_lband_indicator()

    # plt.plot(df["<CLOSE>"])
    # plt.plot(df['kc_kcc'], label='Central KC')
    # plt.plot(df['kc_kch'], label='High KC')
    # plt.plot(df['kc_kcl'], label='Low KC')
    # plt.title('Keltner Channel')
    # plt.legend()
    # plt.show()

    kc_kcc = df['kc_kcc'].to_list()
    kc_kch = df['kc_kch'].to_list()
    kc_kcl = df['kc_kcl'].to_list()



    # __________________________________________ Average true range (ATR) __________________________________________

    # Initialize Average true range Indicator
    indicator_atr = my_general.ta.volatility.AverageTrueRange(high=df["<HIGH>"],
                                                          low=df["<LOW>"],
                                                          close=df["<CLOSE>"], n=20, fillna=True)

    # Add ATR indicator
    df['atr_i'] = indicator_atr.average_true_range()

    # plt.plot(df["<CLOSE>"])
    # plt.plot(df['atr_i'], label='ATR')
    # plt.title('Average true range (ATR)')
    # plt.legend()
    # plt.show()

    atr_i = df['atr_i'].to_list()



    # __________________________________________ Donchian Channel __________________________________________

    # Initialize Donchian Channel Indicator
    indicator_dc = my_general.ta.volatility.DonchianChannel(close=df["<CLOSE>"], n=20, fillna=True)

    # Add Donchian Channel features
    df['dc_dch'] = indicator_dc.donchian_channel_hband()
    df['dc_dcl'] = indicator_dc.donchian_channel_lband()

    # Add Donchian Channel high indicator
    df['dc_dchi'] = indicator_dc.donchian_channel_hband_indicator()

    # Add Donchian Channel low indicator
    df['dc_dcli'] = indicator_dc.donchian_channel_lband_indicator()

    # plt.plot(df["<CLOSE>"])
    # plt.plot(df['dc_dch'], label='High DC')
    # plt.plot(df['dc_dcl'], label='Low DC')
    # plt.title('Donchian Channel')
    # plt.legend()
    # plt.show()

    dc_dch = df['dc_dch'].to_list()
    dc_dcl = df['dc_dcl'].to_list()



    # _____________________________________________________________________________________________________
    # __________________________________________ Trend Indicators _________________________________________
    # _____________________________________________________________________________________________________
    # _____________________________ Average Directional Movement Index (ADX) ________________________________

    # Initialize ADX Indicator
    indicator_adx = my_general.ta.trend.ADXIndicator(high=df["<HIGH>"],
                                                 low=df["<LOW>"],
                                                 close=df["<CLOSE>"], n=20, fillna=True)

    # Add ADX features
    df['adx_aver'] = indicator_adx.adx()
    df['adx_DI_pos'] = indicator_adx.adx_pos()
    df['adx_DI_neg'] = indicator_adx.adx_neg()

    # plt.plot(df["<CLOSE>"])
    # plt.plot(df['adx_aver'], label='ADX')
    # plt.plot(df['adx_DI_pos'], label='+DI')
    # plt.plot(df['adx_DI_neg'], label='-DI')
    # plt.title('ADX')
    # plt.legend()
    # plt.show()

    adx_aver = df['adx_aver'].to_list()
    adx_DI_pos = df['adx_DI_pos'].to_list()
    adx_DI_neg = df['adx_DI_neg'].to_list()



    # _____________________________ Aroon Indicator ________________________________

    # Initialize ADX Indicator
    indicator_ai = my_general.ta.trend.AroonIndicator(close=df["<CLOSE>"], n=20, fillna=True)

    # Add ADX features
    df['ai_i'] = indicator_ai.aroon_indicator()
    df['ai_up'] = indicator_ai.aroon_up()
    df['ai_down'] = indicator_ai.aroon_down()

    # plt.plot(df["<CLOSE>"])
    # plt.plot(df['ai_i'], label='Aroon Indicator')
    # plt.plot(df['ai_up'], label='Aroon Up Channel')
    # plt.plot(df['ai_down'], label='Aroon Down Channel')
    # plt.title('Aroon Indicator')
    # plt.legend()
    # plt.show()

    ai_i = df['ai_i'].to_list()
    ai_up = df['ai_up'].to_list()
    ai_down = df['ai_down'].to_list()



    # _____________________________ Commodity Channel Index (CCI) ________________________________

    # Initialize ADX Indicator
    indicator_ccl = my_general.ta.trend.CCIIndicator(high=df["<HIGH>"],
                                                 low=df["<LOW>"],
                                                 close=df["<CLOSE>"], n=20, c=5, fillna=True)

    # Add ADX features
    df['ccl_i'] = indicator_ccl.cci()

    # # plt.plot(df["<CLOSE>"])
    # plt.plot(df['ccl_i'], label='CCI')
    # plt.title('Commodity Channel Index (CCI)')
    # plt.legend()
    # plt.show()

    ccl_i = df['ccl_i'].to_list()



    # _____________________________ Detrended Price Oscillator (DPO) ________________________________

    # Initialize DPO Indicator
    indicator_dpo = my_general.ta.trend.DPOIndicator(close=df["<CLOSE>"], n=20, fillna=True)

    # Add DPO features
    df['dpo_i'] = indicator_dpo.dpo()

    # plt.plot(df['dpo_i'], label='DPO')
    # plt.title('Detrended Price Oscillator (DPO)')
    # plt.legend()
    # plt.show()

    dpo_i = df['dpo_i'].to_list()



    # _____________________________ Exponential Moving Average (EMA) ________________________________

    # Initialize EMA Indicator
    indicator_ema = my_general.ta.trend.EMAIndicator(close=df["<CLOSE>"], n=20, fillna=True)

    # Add EMA features
    df['ema_i'] = indicator_ema.ema_indicator()

    # plt.plot(df["<CLOSE>"])
    # plt.plot(df['ema_i'], label='EMA')
    # plt.title('Exponential Moving Average (EMA)')
    # plt.legend()
    # plt.show()

    ema_i = df['ema_i'].to_list()



    # _____________________________ Ichimoku Kinkō Hyō (Ichimoku) ________________________________

    # Initialize Ichimoku Indicator
    indicator_ichimoku = my_general.ta.trend.IchimokuIndicator(high=df["<HIGH>"],
                                                           low=df["<LOW>"], n1=10, n2=20, n3=30, visual=False, fillna=True)

    # Add Ichimoku features
    df['ichimoku_a'] = indicator_ichimoku.ichimoku_a()
    df['ichimoku_b'] = indicator_ichimoku.ichimoku_b()

    # plt.plot(df["<CLOSE>"])
    # plt.plot(df['ichimoku_a'], label='Senkou Span A (Leading Span A)')
    # plt.plot(df['ichimoku_b'], label='Senkou Span B (Leading Span B)')
    # plt.title('Ichimoku Kinkō Hyō (Ichimoku)')
    # plt.legend()
    # plt.show()

    ichimoku_a = df['ichimoku_a'].to_list()
    ichimoku_b = df['ichimoku_b'].to_list()



    # _____________________________ KST Oscillator (KST Signal) ________________________________

    # Initialize KST Indicator
    indicator_kst = my_general.ta.trend.KSTIndicator(close=df["<CLOSE>"], r1=10, r2=20, r3=30, r4=40,
                                                 n1=10, n2=10, n3=10, n4=15, nsig=9, fillna=True)
    # Add KST features
    df['kst'] = indicator_kst.kst()
    df['kst_diff'] = indicator_kst.kst_diff()
    df['kst_sig'] = indicator_kst.kst_sig()

    # plt.plot(df["<CLOSE>"])
    # plt.plot(df['kst'], label='Know Sure Thing (KST)')
    # plt.plot(df['kst_diff'], label='Diff Know Sure Thing (KST)')
    # plt.plot(df['kst_sig'], label='Signal Line Know Sure Thing (KST)')
    # plt.title('KST Oscillator (KST Signal)')
    # plt.legend()
    # plt.show()

    kst = df['kst'].to_list()
    kst_diff = df['kst_diff'].to_list()
    kst_sig = df['kst_sig'].to_list()



    # _____________________________ Moving Average Convergence Divergence (MACD) ________________________________

    # Initialize MACD Indicator
    indicator_macd = my_general.ta.trend.MACD(close=df["<CLOSE>"], n_fast=26, n_slow=12, n_sign=9, fillna=True)
    # Add MACD features
    df['macd'] = indicator_macd.macd()
    df['macd_diff'] = indicator_macd.macd_diff()
    df['macd_sig'] = indicator_macd.macd_signal()

    # plt.plot(df["<CLOSE>"])
    # plt.plot(df['macd'], label='MACD Line')
    # plt.plot(df['macd_diff'], label='MACD Histogram')
    # plt.plot(df['macd_sig'], label='Signal Line')
    # plt.title('Moving Average Convergence Divergence (MACD)')
    # plt.legend()
    # plt.show()

    macd = df['macd'].to_list()
    macd_diff = df['macd_diff'].to_list()
    macd_sig = df['macd_sig'].to_list()



    # # _____________________________ Mass Index (MI) ________________________________

    # Initialize MI Indicator
    indicator_mi = my_general.ta.trend.MassIndex(high=df["<HIGH>"], low=df["<LOW>"], n=10, n2=20, fillna=True)
    # Add MI features
    df['mi'] = indicator_mi.mass_index()

    # # plt.plot(df["<CLOSE>"])
    # plt.plot(df['mi'], label='Mass Index (MI)')
    # plt.title('Mass Index (MI)')
    # plt.legend()
    # plt.show()

    mi = df['mi'].to_list()


    # _____________________________ Parabolic Stop and Reverse (Parabolic SAR) ________________________________

    # Initialize PSAR Indicator
    indicator_psar = my_general.ta.trend.PSARIndicator(high=df["<HIGH>"],
                                                   low=df["<LOW>"],
                                                   close=df["<CLOSE>"], step=0.02, max_step=0.2)

    # Add PSAR features
    df['psar_i'] = indicator_psar.psar()
    df['psar_up'] = indicator_psar.psar_up()
    df['psar_down'] = indicator_psar.psar_down()

    df['psar_up_i'] = indicator_psar.psar_up_indicator()
    df['psar_down_i'] = indicator_psar.psar_down_indicator()

    # plt.plot(df["<CLOSE>"])
    # plt.plot(df['psar_i'], label='PSAR value')
    # plt.plot(df['psar_up'], label='PSAR up trend value')
    # plt.plot(df['psar_down'], label='PSAR down trend value')
    # plt.title('Parabolic Stop and Reverse (Parabolic SAR)')
    # plt.legend()
    # plt.show()

    psar_i = df['psar_i'].to_list()
    psar_up = df['psar_up'].to_list()
    psar_down = df['psar_down'].to_list()



    # _____________________________ Trix (TRIX) ________________________________

    # Initialize TRIX Indicator
    indicator_trix = my_general.ta.trend.TRIXIndicator(close=df["<CLOSE>"], n=15, fillna=True)

    # Add TRIX features
    df['trix_i'] = indicator_trix.trix()

    # # plt.plot(df["<CLOSE>"])
    # plt.plot(df['trix_i'], label='TRIX')
    # plt.title('Trix (TRIX)')
    # plt.legend()
    # plt.show()

    trix_i = df['trix_i'].to_list()



    # _____________________________ Vortex Indicator (VI) ________________________________

    # Initialize VI Indicator
    indicator_vi = my_general.ta.trend.VortexIndicator(high=df["<HIGH>"],
                                                   low=df["<LOW>"],
                                                   close=df["<CLOSE>"], n=15, fillna=True)

    # Add VI features
    df['vi_diff'] = indicator_vi.vortex_indicator_diff()
    df['vi_neg'] = indicator_vi.vortex_indicator_neg()
    df['vi_pos'] = indicator_vi.vortex_indicator_pos()

    # # plt.plot(df["<CLOSE>"])
    # plt.plot(df['vi_diff'], label='Diff VI')
    # plt.plot(df['vi_neg'], label='-VI')
    # plt.plot(df['vi_pos'], label='+VI')
    # plt.title('Vortex Indicator (VI)')
    # plt.legend()
    # plt.show()

    vi_diff = df['vi_diff'].to_list()
    vi_neg = df['vi_neg'].to_list()
    vi_pos = df['vi_pos'].to_list()



    # _____________________________________________________________________________________________________
    # _________________________________________ Others Indicators _________________________________________
    # _____________________________________________________________________________________________________
    # ______________________________________ Cumulative Return (CR) _______________________________________

    # Initialize CR Indicator
    indicator_cr = my_general.ta.others.CumulativeReturnIndicator(close=df["<CLOSE>"], fillna=True)

    # Add CR features
    df['cr_i'] = indicator_cr.cumulative_return()

    # plt.plot(df['cr_i'], label='Cumulative Return (CR)')
    # plt.title('Cumulative Return (CR)')
    # plt.legend()
    # plt.show()

    cr_i = df['cr_i'].to_list()



    # # ______________________________________ Daily Log Return (DLR) _______________________________________
    #
    # Initialize DLR Indicator
    indicator_dlr = my_general.ta.others.DailyLogReturnIndicator(close=df["<CLOSE>"], fillna=True)

    # Add DLR features
    df['dlr_i'] = indicator_dlr.daily_log_return()

    # plt.plot(df['dlr_i'], label='Daily Return (DR)')
    # plt.title('Daily Log Return (DLR)')
    # plt.legend()
    # plt.show()

    dlr_i = df['dlr_i'].to_list()



    # _____________________________________________________________________________________________________
    # _________________________________________ Volume Indicators _________________________________________
    # _____________________________________________________________________________________________________
    # ______________________________ Accumulation/Distribution Index (ADI) ________________________________
    #
    # Initialize ADI Indicator
    indicator_adi = my_general.ta.volume.AccDistIndexIndicator(high=df["<HIGH>"],
                                                           low=df["<LOW>"],
                                                           close=df["<CLOSE>"], volume=df["<VOL>"], fillna=True)

    # Add ADI features
    df['adi_i'] = indicator_adi.acc_dist_index()

    # plt.plot(df['adi_i'], label='Accumulation/Distribution Index (ADI)')
    # plt.title('Accumulation/Distribution Index (ADI)')
    # plt.legend()
    # plt.show()

    adi_i = df['adi_i'].to_list()



    # ______________________________ Chaikin Money Flow (CMF) ________________________________
    #
    # Initialize CMF Indicator
    indicator_cmf = my_general.ta.volume.ChaikinMoneyFlowIndicator(high=df["<HIGH>"],
                                                               low=df["<LOW>"],
                                                               close=df["<CLOSE>"],
                                                               volume=df["<VOL>"], n=20, fillna=True)

    # Add CMF features
    df['cmf_i'] = indicator_cmf.chaikin_money_flow()

    # plt.plot(df['cmf_i'], label='CMF')
    # plt.title('Chaikin Money Flow (CMF)')
    # plt.legend()
    # plt.show()

    cmf_i = df['cmf_i'].to_list()


    # ______________________________ Ease of movement (EoM, EMV) ________________________________

    # Initialize (EoM, EMV) Indicator
    indicator_cmf = my_general.ta.volume.EaseOfMovementIndicator(high=df["<HIGH>"],
                                                             low=df["<LOW>"],
                                                             volume=df["<VOL>"], n=20, fillna=True)

    # Add (EoM, EMV) features
    df['cmf_i'] = indicator_cmf.ease_of_movement()
    df['cmf_signal'] = indicator_cmf.sma_ease_of_movement()

    # # plt.plot(df["<CLOSE>"])
    # plt.plot(df['cmf_i'], label='Ease of movement (EoM, EMV)')
    # plt.plot(df['cmf_signal'], label='Signal Ease of movement (EoM, EMV)')
    # plt.title('Ease of movement (EoM, EMV)')
    # plt.legend()
    # plt.show()

    cmf_i = df['cmf_i'].to_list()
    cmf_signal = df['cmf_signal'].to_list()



    # ______________________________ Force Index (FI) ________________________________
    #
    # Initialize FI Indicator
    indicator_fi = my_general.ta.volume.ForceIndexIndicator(close=df["<CLOSE>"],
                                                        volume=df["<VOL>"], n=20, fillna=True)

    # Add FI features
    df['fi_i'] = indicator_fi.force_index()

    # # plt.plot(df["<CLOSE>"])
    # plt.plot(df['fi_i'], label='FI')
    # plt.title('Force Index (FI)')
    # plt.legend()
    # plt.show()

    fi_i = df['fi_i'].to_list()



    # ______________________________ Negative Volume Index (NVI) ________________________________
    #
    # Initialize NVI Indicator
    indicator_nvi = my_general.ta.volume.NegativeVolumeIndexIndicator(close=df["<CLOSE>"],
                                                                  volume=df["<VOL>"], fillna=True)

    # Add NVI features
    df['nvi_i'] = indicator_nvi.negative_volume_index()

    # plt.plot(df["<CLOSE>"])
    # plt.plot(df['nvi_i'], label='NVI')
    # plt.title('Negative Volume Index (NVI)')
    # plt.legend()
    # plt.show()

    nvi_i = df['nvi_i'].to_list()



    # ______________________________ On-balance volume (OBV) ________________________________
    #
    # Initialize OBV Indicator
    indicator_obv = my_general.ta.volume.OnBalanceVolumeIndicator(close=df["<CLOSE>"],
                                                              volume=df["<VOL>"], fillna=True)

    # Add OBV features
    df['obv_i'] = indicator_obv.on_balance_volume()

    # # plt.plot(df["<CLOSE>"])
    # plt.plot(df['obv_i'], label='OBV')
    # plt.title('On-balance volume (OBV)')
    # plt.legend()
    # plt.show()

    obv_i = df['obv_i'].to_list()



    # ______________________________ Volume-price trend (VPT) ________________________________
    #
    # Initialize VPT Indicator
    indicator_vpt = my_general.ta.volume.VolumePriceTrendIndicator(close=df["<CLOSE>"],
                                                               volume=df["<VOL>"], fillna=True)

    # Add VPT features
    df['vpt_i'] = indicator_vpt.volume_price_trend()

    # # plt.plot(df["<CLOSE>"])
    # plt.plot(df['vpt_i'], label='VPT')
    # plt.title('Volume-price trend (VPT)')
    # plt.legend()
    # plt.show()

    vpt_i = df['vpt_i'].to_list()



    # _____________________________________________________________________________________________________
    # ________________________________________ Momentum Indicators ________________________________________
    # _____________________________________________________________________________________________________
    # _________________________________________ Awesome Oscillator ________________________________________
    #
    # Initialize Awesome Oscillator Indicator
    indicator_ao = my_general.ta.momentum.AwesomeOscillatorIndicator(high=df["<HIGH>"],
                                                                 low=df["<LOW>"],
                                                                 s=5, len=34, fillna=True)

    # Add Awesome Oscillator features
    df['ao_i'] = indicator_ao.ao()

    # plt.plot(df['ao_i'], label='AO')
    # plt.title('Awesome Oscillator')
    # plt.legend()
    # plt.show()

    ao_i = df['ao_i'].to_list()



    # ________________________________ Kaufman’s Adaptive Moving Average (KAMA) __________________________________
    # #
    # Initialize KAMA Indicator
    indicator_kama = my_general.ta.momentum.KAMAIndicator(close=df["<CLOSE>"],
                                                      n=10,
                                                      pow1=2, pow2=30, fillna=True)

    # Add KAMA features
    df['kama_i'] = indicator_kama.kama()

    # plt.plot(df["<CLOSE>"])
    # plt.plot(df['kama_i'], label='KAMA')
    # plt.title('Kaufman’s Adaptive Moving Average (KAMA)')
    # plt.legend()
    # plt.show()

    kama_i = df['kama_i'].to_list()



    # ________________________________ Money Flow Index (MFI) __________________________________
    #
    # Initialize MFI Indicator
    indicator_mfi = my_general.ta.momentum.MFIIndicator(high=df["<HIGH>"],
                                                    low=df["<LOW>"],
                                                    close=df["<CLOSE>"],
                                                    volume=df["<VOL>"],
                                                    n=14, fillna=True)

    # Add MFI features
    df['mfi_i'] = indicator_mfi.money_flow_index()

    # # plt.plot(df["<CLOSE>"])
    # plt.plot(df['mfi_i'], label='MFI')
    # plt.title('Money Flow Index (MFI)')
    # plt.legend()
    # plt.show()

    mfi_i = df['mfi_i'].to_list()



    # ________________________________ Rate of Change (ROC) __________________________________
    #
    # Initialize ROC Indicator
    indicator_roc = my_general.ta.momentum.ROCIndicator(close=df["<CLOSE>"],
                                                    n=12, fillna=True)

    # Add ROC features
    df['roc_i'] = indicator_roc.roc()

    # # plt.plot(df["<CLOSE>"])
    # plt.plot(df['roc_i'], label='ROC')
    # plt.title('Rate of Change (ROC)')
    # plt.legend()
    # plt.show()

    roc_i = df['roc_i'].to_list()



    # ________________________________ Relative Strength Index (RSI) __________________________________
    #
    # Initialize RSI Indicator
    indicator_rsi = my_general.ta.momentum.RSIIndicator(close=df["<CLOSE>"],
                                                    n=12, fillna=True)

    # Add RSI features
    df['rsi_i'] = indicator_rsi.rsi()

    # # plt.plot(df["<CLOSE>"])
    # plt.plot(df['rsi_i'], label='RSI')
    # plt.title('Relative Strength Index (RSI)')
    # plt.legend()
    # plt.show()

    rsi_i = df['rsi_i'].to_list()



    # ________________________________ Stochastic Oscillator __________________________________
    # Initialize RSI Indicator
    indicator_stoch = my_general.ta.momentum.StochasticOscillator(high=df["<HIGH>"],
                                                              low=df["<LOW>"],
                                                              close=df["<CLOSE>"],
                                                              n=14, fillna=True)

    # Add RSI features
    df['stoch_i'] = indicator_stoch.stoch()
    df['stoch_signal'] = indicator_stoch.stoch_signal()

    # # plt.plot(df["<CLOSE>"])
    # plt.plot(df['stoch_i'], label='Stochastic Oscillator')
    # plt.plot(df['stoch_signal'], label='Signal Stochastic Oscillator')
    # plt.title('Stochastic Oscillator')
    # plt.legend()
    # plt.show()

    stoch_i = df['stoch_i'].to_list()
    stoch_signal = df['stoch_signal'].to_list()



    # ________________________________ True strength index (TSI) __________________________________
    #
    # Initialize TSI Indicator
    indicator_tsi = my_general.ta.momentum.TSIIndicator(close=df["<CLOSE>"],
                                                    r=25, s=13, fillna=True)

    # Add TSI features
    df['tsi_i'] = indicator_tsi.tsi()

    # # plt.plot(df["<CLOSE>"])
    # plt.plot(df['tsi_i'], label='TSI')
    # plt.title('True strength index (TSI)')
    # plt.legend()
    # plt.show()

    tsi_i = df['tsi_i'].to_list()



    # ________________________________ Ultimate Oscillator __________________________________
    # Initialize Ultimate Oscillator Indicator
    indicator_uo = my_general.ta.momentum.UltimateOscillator(high=df["<HIGH>"],
                                                         low=df["<LOW>"],
                                                         close=df["<CLOSE>"],
                                                         s=7,
                                                         m=14, len=28, ws=4.0, wm=2.0, wl=1.0, fillna=True)

    # Add Ultimate Oscillator features
    df['uo_i'] = indicator_uo.uo()

    # # plt.plot(df["<CLOSE>"])
    # plt.plot(df['uo_i'], label='UO')
    # plt.title('Ultimate Oscillator')
    # plt.legend()
    # plt.show()

    uo_i = df['uo_i'].to_list()



    # ________________________________ Williams %R __________________________________
    # Initialize Williams Indicator
    indicator_wr = my_general.ta.momentum.WilliamsRIndicator(high=df["<HIGH>"],
                                                         low=df["<LOW>"],
                                                         close=df["<CLOSE>"],
                                                         lbp=14, fillna=True)

    # Add Williams features
    df['wr_i'] = indicator_wr.wr()

    # plt.plot(df["<CLOSE>"])
    # plt.plot(df['wr_i'], label='Williams')
    # plt.title('Williams %R')
    # plt.legend()
    # plt.show()

    wr_i = df['wr_i'].to_list()

    filename = 'result_ta'
    old_list_indicators_target_ticker = my_general.read_data_json(curr_path, filename)

    list_indicators_target_ticker.append({
        "diff_value": (float(old_list_indicators_target_ticker[0]["close_value"]) - float(list_open_value[-1])),
        "open_value": list_open_value[-1],
        "close_value": list_close_value[-1],
        "high_value": list_high_value[-1],
        "low_value": list_low_value[-1],
        "volume_value": list_volume_value[-1],
        "bb_bbh": 0.0 if my_general.np.isnan(bb_bbh[-1]) else bb_bbh[-1], "bb_bbl": 0.0 if my_general.np.isnan(bb_bbl[-1]) else bb_bbl[-1], "bb_bbm": 0.0 if my_general.np.isnan(bb_bbm[-1]) else bb_bbm[-1],
        "kc_kcc": 0.0 if my_general.np.isnan(kc_kcc[-1]) else kc_kcc[-1], "kc_kch": 0.0 if my_general.np.isnan(kc_kch[-1]) else kc_kch[-1], "kc_kcl": 0.0 if my_general.np.isnan(kc_kcl[-1]) else kc_kcl[-1],
        "atr_i": 0.0 if my_general.np.isnan(atr_i[-1]) else atr_i[-1],
        "dc_dch": 0.0 if my_general.np.isnan(dc_dch[-1]) else dc_dch[-1], "dc_dcl": 0.0 if my_general.np.isnan(dc_dcl[-1]) else dc_dcl[-1],
        "adx_aver": 0.0 if my_general.np.isnan(adx_aver[-1]) else adx_aver[-1], "adx_DI_pos": 0.0 if my_general.np.isnan(adx_DI_pos[-1]) else adx_DI_pos[-1], "adx_DI_neg": 0.0 if my_general.np.isnan(adx_DI_neg[-1]) else adx_DI_neg[-1],
        "ai_i": 0.0 if my_general.np.isnan(ai_i[-1]) else ai_i[-1], "ai_up": 0.0 if my_general.np.isnan(ai_up[-1]) else ai_up[-1], "ai_down": 0.0 if my_general.np.isnan(ai_down[-1]) else ai_down[-1],
        "ccl_i": 0.0 if my_general.np.isnan(ccl_i[-1]) else ccl_i[-1],
        "dpo_i": 0.0 if my_general.np.isnan(dpo_i[-1]) else dpo_i[-1],
        "ema_i": 0.0 if my_general.np.isnan(ema_i[-1]) else ema_i[-1],
        "ichimoku_a": 0.0 if my_general.np.isnan(ichimoku_a[-1]) else ichimoku_a[-1], "ichimoku_b": 0.0 if my_general.np.isnan(ichimoku_b[-1]) else ichimoku_b[-1],
        "kst": 0.0 if my_general.np.isnan(kst[-1]) else kst[-1], "kst_diff": 0.0 if my_general.np.isnan(kst_diff[-1]) else kst_diff[-1], "kst_sig": 0.0 if my_general.np.isnan(kst_sig[-1]) else kst_sig[-1],
        "macd": 0.0 if my_general.np.isnan(macd[-1]) else macd[-1], "macd_diff": 0.0 if my_general.np.isnan(macd_diff[-1]) else macd_diff[-1], "macd_sig": 0.0 if my_general.np.isnan(macd_sig[-1]) else macd_sig[-1],
        "mi": 0.0 if my_general.np.isnan(mi[-1]) else mi[-1],
        "psar_i": 0.0 if my_general.np.isnan(psar_i[-1]) else psar_i[-1], "psar_up": 0.0 if my_general.np.isnan(psar_up[-1]) else psar_up[-1], "psar_down": 0.0 if my_general.np.isnan(psar_down[-1]) else psar_down[-1],
        "trix_i": 0.0 if my_general.np.isnan(trix_i[-1]) else trix_i[-1],
        "vi_diff": 0.0 if my_general.np.isnan(vi_diff[-1]) else vi_diff[-1], "vi_neg": 0.0 if my_general.np.isnan(vi_neg[-1]) else vi_neg[-1], "vi_pos": 0.0 if my_general.np.isnan(vi_pos[-1]) else vi_pos[-1],
        "cr_i": 0.0 if my_general.np.isnan(cr_i[-1]) else cr_i[-1],
        "dlr_i": 0.0 if my_general.np.isnan(dlr_i[-1]) else dlr_i[-1],
        "adi_i": 0.0 if my_general.np.isnan(adi_i[-1]) else adi_i[-1],
        "cmf_i": 0.0 if my_general.np.isnan(cmf_i[-1]) else cmf_i[-1], "cmf_signal": 0.0 if my_general.np.isnan(cmf_signal[-1]) else cmf_signal[-1],
        "fi_i": 0.0 if my_general.np.isnan(fi_i[-1]) else fi_i[-1],
        "nvi_i": 0.0 if my_general.np.isnan(nvi_i[-1]) else nvi_i[-1],
        "obv_i": 0.0 if my_general.np.isnan(obv_i[-1]) else obv_i[-1],
        "ao_i": 0.0 if my_general.np.isnan(ao_i[-1]) else ao_i[-1],
        "vpt_i": 0.0 if my_general.np.isnan(vpt_i[-1]) else vpt_i[-1],
        "kama_i": 0.0 if my_general.np.isnan(kama_i[-1]) else kama_i[-1],
        "mfi_i": 0.0 if my_general.np.isnan(mfi_i[-1]) else mfi_i[-1],
        "roc_i": 0.0 if my_general.np.isnan(roc_i[-1]) else roc_i[-1],
        "rsi_i": 0.0 if my_general.np.isnan(rsi_i[-1]) else rsi_i[-1],
        "tsi_i": 0.0 if my_general.np.isnan(tsi_i[-1]) else tsi_i[-1],
        "stoch_i": 0.0 if my_general.np.isnan(stoch_i[-1]) else stoch_i[-1], "stoch_signal": 0.0 if my_general.np.isnan(stoch_signal[-1]) else stoch_signal[-1],
        "uo_i": 0.0 if my_general.np.isnan(uo_i[-1]) else uo_i[-1],
        "wr_i": 0.0 if my_general.np.isnan(wr_i[-1]) else wr_i[-1]
    })

    file_name_ta = 'result_ta'
    my_general.write_data_json(list_indicators_target_ticker, curr_path, file_name_ta)

    # _________________________________________________________________________________

    # Check on repeat
    hash_result_ta = my_general.read_data_json(curr_path, 'hash_result_ta')

    file_name = 'result_ta'
    new_hash = my_general.md5(curr_path + file_name + '.json')

    if new_hash == hash_result_ta[0]["hash"]:
        print("___ No the new TA values ___")
        return

    hash_result_ta = [{"hash": new_hash}]

    file_name = 'hash_result_ta'
    my_general.write_data_json(hash_result_ta, curr_path, file_name)

    # _________________________________________________________________________________


if __name__ == '__main__':
    my_general.logging.basicConfig(level=my_general.logging.DEBUG)
    main()
