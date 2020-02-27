# coding: UTF-8

from threading import Thread
from openpyxl import Workbook
import matplotlib.pyplot as plt
import datetime
import logging
import time
import json
import csv
import os


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

        time.sleep(20 * 60)  # sec


def call_stocks(path_name_ta_stocks, path_name_parser_stocks):

    while (datetime.datetime.now().minute > 0) and (datetime.datetime.now().minute < 60):

        th_3 = Thread(target=call_script, args=(path_name_ta_stocks,))
        th_4 = Thread(target=call_script, args=(path_name_parser_stocks,))

        th_3.start()
        th_4.start()

        th_3.join()
        th_4.join()

        time.sleep(10 * 60)  # sec


def main():

    while (datetime.datetime.now().hour > 9) and (datetime.datetime.now().hour < 23):

        th_01 = Thread(target=call_classifiers, args=(path_name_class_e_n, path_name_class_p_n,))
        th_02 = Thread(target=call_stocks, args=(path_name_ta_stocks, path_name_parser_stocks))

        th_01.start()
        th_02.start()

        th_01.join()
        th_02.join()

    else:
        print("Sleep...")


if __name__ == '__main__':
    main()
