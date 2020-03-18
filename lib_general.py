# coding: UTF-8

from finam.export import Exporter, Market, LookupComparator

import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from openpyxl import Workbook
from threading import Thread
from fuzzywuzzy import fuzz
import pandas as pd
import numpy as np
import pymorphy2
import requests
import datetime
import logging
import hashlib
import random
import time
import xlrd
import json
import csv
import sys
import os
import re
import ta


name_ticker = ''


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


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


def convert_csv_to_xls():
    wb = Workbook()
    ws = wb.active
    with open(file_name + extension, 'r') as f:
        for row in csv.reader(f):
            ws.append(row)
    wb.save(file_name + '.xlsx')


def convert_json_to_xlsx():
    path = root_path + 'Helper\\Classifier_politics_news\\'
    file_name = 'params'
    from_extension = '.json'
    to_extension = '.xlsx'

    pd.read_json(path + file_name + from_extension, encoding="utf-8").to_excel(path + file_name + to_extension,
                                                                               encoding="utf-8")

# ______________________________ NN ______________________________


def list_true_value(list_values_to_nn):
    list_diff_values = []
    prev_value = list_values_to_nn[0]
    for idx in range(1, len(list_values_to_nn)):
        if list_values_to_nn[idx] > prev_value: # prev_value + 3%
            list_diff_values.append(1)

        if list_values_to_nn[idx] < prev_value: # prev_value + 3%
            list_diff_values.append(-1)

        if list_values_to_nn[idx] == prev_value:
            list_diff_values.append(0)

        prev_value = list_values_to_nn[idx]

    return list_diff_values


