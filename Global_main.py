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

global prediction_e_n
global prediction_p_n
global market
global result_ta


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


def call_script(path_name_class):
    exec_full(path_name_class)


def call_classifiers(path_name_class_e_n, path_name_class_p_n):

    while (datetime.datetime.now().minute > 0) and (datetime.datetime.now().minute < 60):

        th_1 = Thread(target=call_script, args=(path_name_class_e_n,))
        th_2 = Thread(target=call_script, args=(path_name_class_p_n,))

        th_1.start()
        th_2.start()

        th_1.join()
        th_2.join()

        path = 'Helper\\Classifier_economics_news\\'
        filename = 'prediction_e_n'
        prediction_e_n = read_data_json(root_path + path, filename)
        print(prediction_e_n)

        path = 'Helper\\Classifier_politics_news\\'
        filename = 'prediction_p_n'
        prediction_p_n = read_data_json(root_path + path, filename)
        print(prediction_p_n)

        time.sleep(20 * 60)  # sec


def call_stocks(path_name_ta_stocks, path_name_parser_stocks):

    while (datetime.datetime.now().minute > 0) and (datetime.datetime.now().minute < 60):

        th_3 = Thread(target=call_script, args=(path_name_ta_stocks,))
        th_4 = Thread(target=call_script, args=(path_name_parser_stocks,))

        th_3.start()
        th_4.start()

        th_3.join()
        th_4.join()

        path = 'Helper\\TA_stocks\\'
        filename = 'result_ta'
        market = read_data_json(root_path + path, filename)

        path = 'Helper\\Parser_stocks\\'
        filename = 'market'
        result_ta = read_data_json(root_path + path, filename)

        time.sleep(10 * 60)  # sec


def main():
    app = HBoxLayoutExample()
    app.run()

    while (datetime.datetime.now().hour > 9) and (datetime.datetime.now().hour < 23):

        th_01 = Thread(target=call_classifiers, args=(path_name_class_e_n, path_name_class_p_n,))
        th_02 = Thread(target=call_stocks, args=(path_name_ta_stocks, path_name_parser_stocks))

        th_01.start()
        th_02.start()

        print("Result ->>>")
        print(prediction_e_n)
        print(prediction_p_n)
        print(market)
        print(result_ta)

        th_01.join()
        th_02.join()

    else:
        print("Sleep...")


if __name__ == '__main__':
    main()
