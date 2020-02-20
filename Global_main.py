from openpyxl import Workbook
import matplotlib.pyplot as plt
import datetime
import logging
import json
import csv


SYMBOLS = ['FXRB',
           'FXMM'
           'FXRU',
           'FXRB',
           'FXWO',
           'FXWR'
           ]

start = datetime.datetime(2020, 1, 1);
end = datetime.datetime(datetime.datetime.now().year,
                        datetime.datetime.now().month,
                        datetime.datetime.now().day);



file_name_class_e_n = 'Classifier_economics_news\\Classifier_e_n.py'
file_name_class_p_n = 'Classifier_politics_news\\Classifier_p_n.py'
# file_name_class_e_n = 'Classifier_economics_news\\Classifier_e_n.py'
# file_name_class_e_n = 'Classifier_economics_news\\Classifier_e_n.py'


def exec_full(filepath):
    global_namespace = {
        "__file__": filepath,
        "__name__": "__main__",
    }
    with open(filepath, 'rb') as file:
        exec(compile(file.read(), filepath, 'exec'), global_namespace)


def main():

    exec_full(file_name_class_e_n)

if __name__ == '__main__':
    main()
