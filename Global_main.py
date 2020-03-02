# coding: UTF-8



from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.app import App

from threading import Thread
from openpyxl import Workbook
import matplotlib.pyplot as plt
import datetime
import logging
import random
import time
import json
import csv
import os


from finam.export import Exporter, Market, LookupComparator
from keras.layers import Dense, Dropout
from keras.models import Sequential

import numpy as np
import pandas as pd
import requests
import keras




red = [1, 0, 0, 1]
green = [0, 1, 0, 1]
blue = [0, 0, 1, 1]
purple = [1, 0, 1, 1]

class HBoxLayoutExample(App):
    def build(self):
        layout = BoxLayout(padding=10)
        colors = [red, green, blue, purple]

        button = Button(text='Hello from Kivy',
                        background_color=green)
        button.bind(on_press=self.on_press_button)
        layout.add_widget(button)
        for i in range(5):
            btn = Button(text="Button #%s" % (i + 1),
                         background_color=red
                         )

            layout.add_widget(btn)
        return layout

    def on_press_button(self, instance):
        print('Вы нажали на кнопку!')


root_path = 'C:\\Users\\user\\0_Py\\'

SYMBOLS = ['FXRB',
           'FXMM',
           'FXRU',
           'FXRB',
           'FXWO',
           'FXWR'
           ]

start = datetime.datetime(2020, 1, 1);
end = datetime.datetime(datetime.datetime.now().year,
                        datetime.datetime.now().month,
                        datetime.datetime.now().day);


path_name_class_e_n = 'Classifier_economics_news\\Classifier_e_n.py'
path_name_class_p_n = 'Classifier_politics_news\\Classifier_p_n.py'
path_name_ta_stocks = 'TA_stocks\\TA_stocks.py'
path_name_parser_stocks = 'Parser_stocks\\Parser_stocks.py'

prediction_e_n = []
prediction_p_n = []
market = []
result_ta = []


def write_data_json(data, path, file_name):
    extension = '.json'

    with open(path + file_name + extension, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def read_data_json(path, file_name):
    extension = '.json'
    data = []

    with open(path + file_name + extension, encoding="utf-8") as json_file:
        data = json.load(json_file)

    return data


def exec_full(file_path):
    global_namespace = {
        "__file__": file_path,
        "__name__": "__main__",
    }
    with open(file_path, 'rb') as file:
        exec(compile(file.read(), file_path, 'exec'), global_namespace)


def main():
    app = HBoxLayoutExample()
    app.run()

    while (datetime.datetime.now().hour > 9) and (datetime.datetime.now().hour < 23):

        # exec_full(path_name_class_e_n)
        # exec_full(path_name_class_p_n)
        # exec_full(path_name_parser_stocks)
        # exec_full(path_name_ta_stocks)


        print("Result ->>>")

        path = 'Helper\\Classifier_economics_news\\'
        filename = 'prediction_e_n'
        prediction_e_n = read_data_json(root_path + path, filename)

        path = 'Helper\\Classifier_politics_news\\'
        filename = 'prediction_p_n'
        prediction_p_n = read_data_json(root_path + path, filename)

        path = 'Helper\\TA_stocks\\'
        filename = 'result_ta'
        result_ta = read_data_json(root_path + path, filename)

        path = 'Helper\\Parser_stocks\\'
        filename = 'market'
        market = read_data_json(root_path + path, filename)

        # print(prediction_e_n)
        # print(prediction_p_n)
        # print(market)
        # print(result_ta)


        # задаем для воспроизводимости результатов
        np.random.seed(2)
        path = 'Helper\\'
        model_name = root_path + path + 'NN_Main_model.h5'

        X = []

        X.append(prediction_e_n['score'])

        X.append(prediction_p_n['score'])

        for ticker in market:
            for input in ticker:
                X.append(input['open_value'])
                X.append(input['close_value'])
                X.append(input['high_value'])
                X.append(input['low_value'])
                X.append(input['volume_value'])


        X.append(result_ta[0]['open_value'])
        X.append(result_ta[0]['close_value'])
        X.append(result_ta[0]['high_value'])
        X.append(result_ta[0]['low_value'])
        X.append(result_ta[0]['volume_value'])
        X.append(result_ta[0]['adi_i'])
        X.append(result_ta[0]['adx_aver'])
        X.append(result_ta[0]['adx_DI_pos'])
        X.append(result_ta[0]['adx_DI_neg'])
        X.append(result_ta[0]['ai_i'])
        X.append(result_ta[0]['ai_up'])
        X.append(result_ta[0]['ai_down'])
        X.append(result_ta[0]['ao_i'])
        X.append(result_ta[0]['atr_i'])
        X.append(result_ta[0]['bb_bbh'])
        X.append(result_ta[0]['bb_bbl'])
        X.append(result_ta[0]['bb_bbm'])
        X.append(result_ta[0]['ccl_i'])
        X.append(result_ta[0]['cmf_i'])
        X.append(result_ta[0]['cmf_signal'])
        X.append(result_ta[0]['cr_i'])

        X.append(result_ta[0]['dc_dch'])
        X.append(result_ta[0]['dc_dcl'])
        X.append(result_ta[0]['dlr_i'])
        X.append(result_ta[0]['dpo_i'])
        X.append(result_ta[0]['ema_i'])
        X.append(result_ta[0]['fi_i'])
        X.append(result_ta[0]['ichimoku_a'])
        X.append(result_ta[0]['ichimoku_b'])
        X.append(result_ta[0]['kama_i'])
        X.append(result_ta[0]['kc_kcc'])
        X.append(result_ta[0]['kc_kch'])
        X.append(result_ta[0]['kc_kcl'])
        X.append(result_ta[0]['kst'])
        X.append(result_ta[0]['kst_diff'])
        X.append(result_ta[0]['kst_sig'])
        X.append(result_ta[0]['vi_diff'])
        X.append(result_ta[0]['vi_neg'])
        X.append(result_ta[0]['vi_pos'])

        X.append(result_ta[0]['mfi_i'])
        X.append(result_ta[0]['mi'])
        X.append(result_ta[0]['nvi_i'])
        X.append(result_ta[0]['obv_i'])
        X.append(result_ta[0]['psar_i'])
        X.append(result_ta[0]['psar_up'])
        X.append(result_ta[0]['psar_down'])
        X.append(result_ta[0]['roc_i'])
        X.append(result_ta[0]['rsi_i'])
        X.append(result_ta[0]['stoch_i'])
        X.append(result_ta[0]['stoch_signal'])
        X.append(result_ta[0]['trix_i'])
        X.append(result_ta[0]['tsi_i'])
        X.append(result_ta[0]['uo_i'])
        X.append(result_ta[0]['vpt_i'])

        count_inputs = len(X)
        # print("Len NN: " + str(count_inputs))

        # создаем модели, добавляем слои один за другим
        model = Sequential()
        model.add(Dense(count_inputs, input_dim=count_inputs, activation='relu'))
        model.add(Dense(count_inputs - 5, activation='relu'))
        model.add(Dense(count_inputs - 10, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(count_inputs - 20, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(count_inputs - 30, activation='sigmoid'))
        model.add(Dense(count_inputs, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))

         # компилируем модель, используем градиентный спуск adam
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])


        # for news in listWordsToNN:
        #     # разбиваем датасет на матрицу параметров (X) и вектор целевой переменной (Y)
        #     one_sentence_news = news.ravel()
        #
        #     X.append(one_sentence_news)
        #
        # X = np.asarray(X, dtype=np.float32)
        # Y = np.asarray(listTrueValue, dtype=np.float32)
        #
        # if os.path.exists(model_name) != False:
        #     # Recreate the exact same model
        #     new_model = keras.models.load_model(model_name)
        # else:
        #     new_model = model
        #
        # # обучаем нейронную сеть
        # history = new_model.fit(X, Y, epochs=500, batch_size=64)
        #
        # # Export the model to a SavedModel
        # new_model.save(model_name)
        #
        # # оцениваем результат
        # scores = new_model.predict(X)
        # print("\n%s: %.2f%%" % (new_model.metrics_names[1], scores[1] * 100))
        # print(scores)
        #
        # main_prediction = {"score": float(scores[-1] * 100)}
        # print(main_prediction)
        #
        # path = root_path + 'Helper\\Classifier_economics_news\\'
        # file_name_prediction = 'main_prediction'
        #
        # write_data_json(main_prediction, path, file_name_prediction)
        #
        # time.sleep(15 * 60)  # sec

    else:
        print("Sleep...")


if __name__ == '__main__':
    main()
