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
# file_name_class_e_n = 'Classifier_economics_news\\Classifier_e_n.py'
# file_name_class_e_n = 'Classifier_economics_news\\Classifier_e_n.py'


def exec_full(file_path):
    global_namespace = {
        "__file__": file_path,
        "__name__": "__main__",
    }
    with open(file_path, 'rb') as file:
        exec(compile(file.read(), file_path, 'exec'), global_namespace)


def call_classifier(path_name_class):
    exec_full(path_name_class)


def main():

    while (datetime.datetime.now().hour > 9) and (datetime.datetime.now().hour < 23):

        while (datetime.datetime.now().minute != 0) or (datetime.datetime.now().minute == 30):
            print("Infinity and Beyond!!!")

            th_1 = Thread(target=call_classifier, args=(path_name_class_e_n,))
            th_2 = Thread(target=call_classifier, args=(path_name_class_p_n,))

            th_1.start()
            th_2.start()

            th_1.join()
            th_2.join()

            time.sleep(5 * 60)  # sec

    else:
        print("Sleep...")

if __name__ == '__main__':
    main()
