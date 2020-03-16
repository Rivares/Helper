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
from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.models import Sequential

import numpy as np
import pandas as pd
import requests
import keras


from finam.export import Exporter, Market, LookupComparator
from keras.models import Sequential
from keras.layers import Dense, Dropout
from openpyxl import Workbook
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
import pymorphy2
import datetime
import requests
import logging

import keras

import xlrd
import csv
import os
import re



from finam.export import Exporter, Market, LookupComparator
from openpyxl import Workbook
import matplotlib.pyplot as plt
import pandas as pd
import datetime

import time

import csv



from finam.export import Exporter, Market, LookupComparator
from openpyxl import Workbook
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import logging
import hashlib
import json
import csv
import ta


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


def read_params_xlsx():
    country_path = root_path + 'Helper\\Classifier_politics_news\\'
    country_file_name = 'params'
    country_extension = '.xlsx'

    workbook = xlrd.open_workbook(country_path + country_file_name + country_extension, on_demand=True)
    worksheet = workbook.sheet_by_index(0)

    if os.stat(country_path + country_file_name + country_extension).st_size != 0:
        first_row = []
        for col in range(worksheet.ncols):
            first_row.append(worksheet.cell_value(0, col))

        listParams_E_N = []
        for row in range(1, worksheet.nrows):
            elm = {}
            for col in range(worksheet.ncols):
                elm[first_row[col]] = worksheet.cell_value(row, col)
            listParams_E_N.append(elm)

        print(listParams_E_N)
    else:
        print("Error read file!")

    return listParams_E_N


def convert_json_to_xlsx():
    path = root_path + 'Helper\\Classifier_politics_news\\'
    file_name = 'params'
    from_extension = '.json'
    to_extension = '.xlsx'

    pd.read_json(path + file_name + from_extension, encoding="utf-8").to_excel(path + file_name + to_extension,
                                                                               encoding="utf-8")

