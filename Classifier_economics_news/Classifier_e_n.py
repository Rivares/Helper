# coding: utf8

import datetime
from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
import pymorphy2
import json
import xlrd
import csv
import os
import re



class Spider(object):
    def __init__(self, title, additionally, href, date):
        """Constructor"""
        self.title = title
        self.additionally = additionally
        self.href = href
        self.date = date

    def get_params(self):
        return self.title, self.additionally, self.href, self.date


'''______________________________________________________________________'''

#
# def similar(a, b):
#     return SequenceMatcher(None, a, b).ratio()


def read_params_xlsx():
    country_path = 'C:\\Users\\user\\0_Py\\Helper\\Classifier_economics_news\\'
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
    path = 'C:\\Users\\user\\0_Py\\Helper\\Classifier_economics_news\\'
    file_name = 'params'
    from_extension = '.json'
    to_extension = '.xlsx'

    pd.read_json(path + file_name + from_extension, encoding="utf-8").to_excel(path + file_name + to_extension,
                                                                                   encoding="utf-8")


def read_params_json():
    path = 'C:\\Users\\user\\0_Py\\Helper\\Classifier_economics_news\\'
    file_name = 'params'
    extension = '.json'

    listParams_E_N = []

    with open(path + file_name + extension, encoding="utf-8") as json_file:
        listParams_E_N = json.load(json_file)

    return listParams_E_N


def write_params_json(listParams_E_N):
    path = 'C:\\Users\\user\\0_Py\\Helper\\Classifier_economics_news\\'
    file_name = 'params_new'
    extension = '.json'

    with open(path + file_name + extension, "w", encoding="utf-8") as json_file:
            json.dump(listParams_E_N, json_file, ensure_ascii=False, indent=4)


def read_article_csv():
    path = '../Parser_economics_news/'
    file_name = 'economics_news'
    extension = '.csv'
    listSpider_E_N = []

    if os.stat(path + file_name + extension).st_size != 0:
        with open(path + file_name + extension, newline='\n') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                # reader.fieldnames[i] - i = 0 - title; 1 - additionally; 2 - href; 3 - date
                if str(row.get(reader.fieldnames[0])) != str(reader.fieldnames[0]):
                    listSpider_E_N.append(
                        Spider(row.pop(reader.fieldnames[0])
                               , row.pop(reader.fieldnames[1])
                               , row.pop(reader.fieldnames[2])
                               , row.pop(reader.fieldnames[3])
                               )
                    )
    else:
        print("Error read file!")
    return listSpider_E_N


def main():

    length_sentence = 20

    # _________________________________________________________________________________

    # Creating list of news + to Lower Case + delete ',' and  '.'

    listSpider_E_N = read_article_csv()
    # print(listSpider_E_N.__len__())

    reg = re.compile('[^а-яА-Я -]')
    print()
    for obj in listSpider_E_N:
        obj.title = obj.title.lower()
        obj.title = reg.sub('', obj.title)
        obj.additionally = obj.additionally.lower()
        obj.additionally = reg.sub('', obj.additionally)
        # print(obj.title, obj.additionally, obj.href, obj.date, sep=' ')

    # _________________________________________________________________________________

    # Normalization the list of news

    morph = pymorphy2.MorphAnalyzer()

    for obj in listSpider_E_N:
        obj.title = (' '.join([morph.normal_forms(w)[0] for w in obj.title.split()]))
        obj.additionally = (' '.join([morph.normal_forms(w)[0] for w in obj.additionally.split()]))

    # _________________________________________________________________________________

    # Read reference words from json file

    # listParams_E_N = read_params_xlsx()
    listParams_E_N = read_params_json()
    # write_params_json(listParams_E_N)
    # convert_json_to_xlsx()

    # _________________________________________________________________________________

    # Normalization reference words and rewrite json file
    #
    # morph = pymorphy2.MorphAnalyzer()
    #
    # newListParams_E_N = []
    # for obj in listParams_E_N:
    #     new_name = ' '.join([morph.normal_forms(w)[0] for w in obj.get('name').split()])
    #     new_synonyms = ' '.join([morph.normal_forms(w)[0] for w in obj.get('synonyms').split()])
    #     params = {'name': new_country, 'synonyms': new_synonyms, 'impact': item.get('impact')}
    #     newListParams_E_N.append(params)
    #
    # write_params_json(newListParams_E_N)
    # listParams_E_N = newListParams_E_N
    #
    # _________________________________________________________________________________

    # Get only text information from title and additionally

    newListSpider_E_N = []
    for obj in listSpider_E_N:
        newListSpider_E_N.append(obj.title + ' ' + obj.additionally)

    listSpider_E_N = newListSpider_E_N

    # _________________________________________________________________________________

    # Transform to array words

    listWords = []
    for obj in listSpider_E_N:
        listWords.append(obj.split())

    print(listWords)

    # _________________________________________________________________________________

    # Finding reference words to array words

    future_weigths = np.zeros((len(pd.Series(np.random.randn(length_sentence)))))
    print(future_weigths)

    cnt = 0
    for item in listWords:
        for obj in item:
            for params in listParams_E_N:
                if fuzz.ratio(params.get('name'), obj.title().lower()) > 90:
                    # print("I found of name! --->>> " + str(obj))
                    future_weigths[cnt] = float(params.get('impact'))
                    break
                else:
                    if len(params.get('synonyms')) > 1:
                        for it in params.get('synonyms'):
                            if fuzz.ratio(str(it), str(obj.title().lower())) > 70:
                                # print("I found of synonyms! --->>> " + str(obj.title().lower()))
                                future_weigths[cnt] = float(params.get('impact'))
                                break
    ++cnt


    print(future_weigths)

if __name__ == '__main__':
    main()
