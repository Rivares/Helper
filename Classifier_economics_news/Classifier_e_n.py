# coding: utf8

from openpyxl import Workbook
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
import pymorphy2
import datetime
import requests
import json
import xlrd
import csv
import os
import re


class Spider(object):
    def __init__(self, title, additionally, href, time):
        """Constructor"""
        self.title = title
        self.additionally = additionally
        self.href = href
        self.time = time

    def get_params(self):
        return self.title, self.additionally, self.href, self.time


'''______________________________________________________________________'''


def get_html(url):
    r = requests.get(url)
    return r.text


def write_article_csv(data):
    with open(file_name + extension, 'a', newline='') as f:
        fieldnames = ['title', 'additionally', 'href', 'time']
        writer = csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'title': data['title'],
                         'additionally': data['additionally'],
                         'href': data['href'],
                         'time': data['time']
                         })

def read_article_csv():
    path = '../Parser_economics_news/'
    file_name = 'economics_news'
    extension = '.csv'
    listSpider_E_N = []

    if os.stat(path + file_name + extension).st_size != 0:
        with open(path + file_name + extension, newline='\n') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                # reader.fieldnames[i] - i = 0 - title; 1 - additionally; 2 - href; 3 - time
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


def convert_csv_to_xls():
    wb = Workbook()
    ws = wb.active
    with open(file_name + extension, 'r') as f:
        for row in csv.reader(f):
            ws.append(row)
    wb.save(file_name + '.xlsx')


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


def get_page_data(html, article_data):
    soup = BeautifulSoup(html, 'lxml')
    divs = soup.find('div', class_='list list-tags')
    ads = divs.find_all('div', class_='list-item', limit=10)

    article_data.clear()
    title = ''
    addit = ''
    href = ''
    time = ''
    for ad in ads:
        try:
            div = ad.find('div', class_='list-item__content').find('a',
                                                                   class_='list-item__title color-font-hover-only');
            title = div.text;

            href = div.get('href');

            title_picture = ad.find('div', class_='list-item__content') \
                .find('a', class_='list-item__image') \
                .find('picture') \
                .find('img') \
                .get('title')
            addit = title_picture;

            time = ad.find('div', class_='list-item__info') \
                .find('div', class_='list-item__date').text;
            time = time.split(' ')
            time = time[1]

            data = {'title': title, 'additionally': addit, 'href': href, 'time': time}
            article_data.append(data)
        except:
            title = 'Error'
            addit = 'Error'
            href = 'Error'
            time = 'Error'

    return article_data


def main():
    base_url = "https://ria.ru/economy/"
    article_data = []

    # os.remove(file_name + '.csv')
    # os.remove(file_name + '.xlsx')
    # os.remove(file_name + '.json')

    url_gen = base_url
    html = get_html(url_gen)
    article_data = get_page_data(html, article_data)

    # print(article_data.__len__())
    path = 'C:\\Users\\user\\0_Py\\Helper\\Parser_economics_news\\'
    file_name = 'economics_news'
    write_data_json(article_data, path, file_name)

    length_sentence = 20

    # _________________________________________________________________________________

    # Creating list of news + to Lower Case + delete ',' and  '.'

    path = 'C:\\Users\\user\\0_Py\\Helper\\Parser_economics_news\\'
    file_name = 'economics_news'
    news = read_data_json(path, file_name)

    listSpider_E_N = []
    for item in news:
        listSpider_E_N.append(Spider(item['title']
                                     , item['additionally']
                                     , item['href']
                                     , item['time']
                                     )
                              )

    # listSpider_E_N = read_article_csv()
    # print(listSpider_E_N.__len__())

    reg = re.compile('[^а-яА-Я -]')

    for obj in listSpider_E_N:
        obj.title = obj.title.lower()
        obj.title = reg.sub('', obj.title)
        obj.additionally = obj.additionally.lower()
        obj.additionally = reg.sub('', obj.additionally)
        # print(obj.title, obj.additionally, obj.href, obj.time, sep=' ')

    # _________________________________________________________________________________

    # Deleting repeats hrefs

    # print(listSpider_E_N[0].title,
    #       listSpider_E_N[0].additionally,
    #       listSpider_E_N[0].href,
    #       listSpider_E_N[0].time,
    #       sep=' ')

    idx_1 = 0
    idx_2 = 0
    for idx_1 in range(1, len(listSpider_E_N) - 1):
        ref_href = listSpider_E_N[idx_1].href
        idx_2 = idx_1 + 1
        for j in range(idx_2, len(listSpider_E_N) - 1):
            if listSpider_E_N[j].href == ref_href:
                listSpider_E_N.remove(listSpider_E_N[j])

    # print(listSpider_E_N[0].title,
    #       listSpider_E_N[0].additionally,
    #       listSpider_E_N[0].href,
    #       listSpider_E_N[0].time,
    #       sep=' ')

    # _________________________________________________________________________________

    # Normalization the list of news

    morph = pymorphy2.MorphAnalyzer()

    for obj in listSpider_E_N:
        obj.title = (' '.join([morph.normal_forms(w)[0] for w in obj.title.split()]))
        obj.additionally = (' '.join([morph.normal_forms(w)[0] for w in obj.additionally.split()]))

    # _________________________________________________________________________________

    # Read reference words from json file

    # listParams_E_N = read_params_xlsx()
    path = 'C:\\Users\\user\\0_Py\\Helper\\Classifier_economics_news\\'
    file_name = 'params'
    listParams_E_N = read_data_json(path, file_name)
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

    # _________________________________________________________________________________

    # Delete to array words

    for sentence in listWords:
        # print(sentence)
        for word in sentence:
            p = morph.parse(word)[0]
            if p.tag.POS == 'PREP':
                sentence.remove(word)
        # print(sentence)
    # _________________________________________________________________________________

    # # Finding reference words to array words
    #
    # # _________________________________________________________________________________
    #
    # # For Real-Time mode
    # #
    # # future_weigths = np.zeros(length_sentence, dtype=float)
    # #
    # # cnt = 0
    # # # for item in listWords:
    # #
    # # header = listWords[1]
    # # print(header)
    # # print(len(header))
    # #
    # # for obj in header:
    # #     # print(obj.lower())
    # #     for params in listParams_E_N:
    # #         if fuzz.ratio(params.get('name'), obj.lower()) > 90:
    # #             # print("I found of name! --->>> " + str(obj))
    # #             future_weigths[cnt] = float(params.get('impact'))
    # #             break
    # #         else:
    # #             if len(params.get('synonyms')) >= 1:
    # #                 for it in params.get('synonyms'):
    # #                     if fuzz.ratio(str(it), str(obj.lower())) > 80:
    # #                         # print("I found of synonyms! --->>> " + str(obj.lower()))
    # #                         future_weigths[cnt] = float(params.get('impact'))
    # #                         break
    # #     cnt = cnt + 1
    # #
    # # _________________________________________________________________________________
    # #
    # For Trainging NN

    # _________________________________________________________________________________

    # future_weigths = np.zeros(length_sentence, dtype=float)
    list_future_weigths = np.zeros((len(listWords), length_sentence), dtype=float)

    idx_word = 0
    idx_sentence = 0
    for header in listWords:
        # print(header)
        for obj in header:
            # print(obj.lower())
            for params in listParams_E_N:
                if fuzz.ratio(params.get('name'), obj.lower()) > 90:
                    # print("I found of name! --->>> " + str(obj))
                    list_future_weigths[idx_sentence][idx_word] = float(params.get('impact'))
                    break
                else:
                    if len(params.get('synonyms')) >= 1:
                        for it in params.get('synonyms'):
                            if fuzz.ratio(str(it), str(obj.lower())) > 80:
                                # print("I found of synonyms! --->>> " + str(obj.lower()))
                                list_future_weigths[idx_sentence][idx_word] = float(params.get('impact'))
                                break
            idx_word = idx_word + 1
        idx_word = 0
        idx_sentence = idx_sentence + 1

    # print(list_future_weigths[len(listWords) - 2])

    # _________________________________________________________________________________

    # Appending feature of applicants to list to json file
    # 1 day for remove from applicants.json
    # 240 it's 50% <- 1 day - 24 hours - 48 query * 10 news
    # 384 it's 80% <- 1 day - 24 hours - 48 query * 10 news
    # 3 day for appending to params.json

    border = 240

    idx_word = 0
    idx_sentence = 0
    for header in listWords:
        # print(header)
        for obj in header:
            if list_future_weigths[idx_sentence][idx_word] == 0:
                path = 'C:\\Users\\user\\0_Py\\Helper\\Classifier_economics_news\\'
                file_name = 'applicants'
                feature_list_applicants = read_data_json(path, file_name)

                # find to feature_list_applicants obj
                success = 0
                # Increase count
                for item in feature_list_applicants:
                    # print(ithem["name"], ithem["count"], sep=' ')
                    if obj == item["name"]:
                        item["count"] = item["count"] + 1
                        print("I found of name! --->>> " + str(item["count"]))
                        path = 'C:\\Users\\user\\0_Py\\Helper\\Classifier_economics_news\\'
                        file_name = 'applicants'
                        write_data_json(feature_list_applicants, path, file_name)
                        success = 1

                        if item["count"] >= border:
                            rng = np.random.default_rng()
                            path = 'C:\\Users\\user\\0_Py\\Helper\\Classifier_economics_news\\'
                            file_name = 'params'
                            list_params = read_data_json(path, file_name)

                            list_params.append({"name": item["name"],
                                                "synonyms": [""],
                                                "impact": (rng.random() - 0.5)
                                                })
                            path = 'C:\\Users\\user\\0_Py\\Helper\\Classifier_economics_news\\'
                            file_name = 'applicants'
                            write_data_json(list_params, path, file_name)
                            feature_list_applicants.remove(item)
                            path = 'C:\\Users\\user\\0_Py\\Helper\\Classifier_economics_news\\'
                            file_name = 'applicants'
                            write_data_json(feature_list_applicants, path, file_name)

                        break
                # Add new feature
                if success == 0:
                    new_feature_applicant = {"name": obj, "count": 1}
                    feature_list_applicants.append(new_feature_applicant)
                    path = 'C:\\Users\\user\\0_Py\\Helper\\Classifier_economics_news\\'
                    file_name = 'applicants'
                    write_data_json(feature_list_applicants, path, file_name)
                    print(obj)

            idx_word = idx_word + 1
        idx_word = 0
        idx_sentence = idx_sentence + 1


    # feature_list_applicants.append()

    # write_applicants_json(feature_list_applicants)

    # _________________________________________________________________________________


if __name__ == '__main__':
    main()
