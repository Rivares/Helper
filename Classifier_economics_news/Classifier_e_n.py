# coding: utf8

from finam.export import Exporter, Market, LookupComparator
from keras.models import Sequential
from keras.layers import Dense
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

# def write_data_csv(data, path, file_name):
#     with open(path + file_name + '.csv', 'w', newline='') as f:
#         fieldnames = []
#         for item in data:
#             fieldnames.append(data.)
#         writer = csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerow({'title': data['title'],
#                          'additionally': data['additionally'],
#                          'href': data['href'],
#                          'date': data['date']
#                          })


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


# ______________________________ Parser ______________________________


def get_html(url):
    r = requests.get(url)
    return r.text


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
                .find('div', class_='list-item__date').text
            time = time.split(', ')
            time = time[-1]

            data = {'title': title, 'additionally': addit, 'href': href, 'time': time}
            article_data.append(data)

        except:
            title = 'Error'
            addit = 'Error'
            href = 'Error'
            time = 'Error'

    return article_data


# ______________________________ NN ______________________________

def list_true_value(list_values_to_nn):
    list_diff_values = []
    prev_value = list_values_to_nn[0]
    for idx in range(1, len(list_values_to_nn)):
        if list_values_to_nn[idx] > prev_value:
            list_diff_values.append(1)

        if list_values_to_nn[idx] < prev_value:
            list_diff_values.append(-1)

        if list_values_to_nn[idx] == prev_value:
            list_diff_values.append(0)

        prev_value = list_values_to_nn[idx]

    return list_diff_values

def sigmoid(x):
    # Функция активации sigmoid:: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # Производная от sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    # y_true и y_pred являются массивами numpy с одинаковой длиной
    return ((y_true - y_pred) ** 2).mean()


class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # Вводные данные о весе, добавление смещения
        # и последующее использование функции активации

        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


class OurNeuralNetwork:
    """
    Нейронная сеть, у которой:
        - 2 входа
        - скрытый слой с двумя нейронами (h1, h2)
        - слой вывода с одним нейроном (o1)

    *** ВАЖНО ***:
    Код ниже написан как простой, образовательный. НЕ оптимальный.
    Настоящий код нейронной сети выглядит не так. НЕ ИСПОЛЬЗУЙТЕ этот код.
    Вместо этого, прочитайте/запустите его, чтобы понять, как работает эта сеть.
    """

    def __init__(self, dim_in, dim_h, count_h):
        # Вес
        # self.list_weights = []
        # for
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Смещения
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        # x является массивом numpy с двумя элементами
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        """
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - all_y_trues is a numpy array with n elements.
            Elements in all_y_trues correspond to those in data.
        """
        learn_rate = 0.1
        epochs = 10000  # количество циклов во всём наборе данных

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # --- Выполняем обратную связь (нам понадобятся эти значения в дальнейшем)
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # --- Подсчет частных производных
                # --- Наименование: d_L_d_w1 представляет "частично L / частично w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Нейрон o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Нейрон h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Нейрон h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # --- Обновляем вес и смещения
                # Нейрон h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Нейрон h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Нейрон o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            # --- Подсчитываем общую потерю в конце каждой фазы
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))


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
    path = 'C:\\Users\\user\\0_Py\\Helper\\Classifier_economics_news\\'
    file_name = 'economics_news'
    write_data_json(article_data, path, file_name)

    count_sentences = article_data.__len__()
    count_words = 30
    count_charters = 30

    # _________________________________________________________________________________

    # Creating list of news + to Lower Case + delete ',' and  '.'

    path = 'C:\\Users\\user\\0_Py\\Helper\\Classifier_economics_news\\'
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
    time_news = []
    for news in listSpider_E_N:
        newListSpider_E_N.append(news.title + ' ' + news.additionally)
        time_news.append(news.time)

    listSpider_E_N = newListSpider_E_N

    # _________________________________________________________________________________

    # Transform to array words

    listWords = []
    for news in listSpider_E_N:
        listWords.append(news.split())

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

    # Transform to digital mode

    # print(listWords[0][0])

    newListWords = []
    listWordsToNN = np.zeros((count_sentences, count_words, count_charters))

    idx_sentence = 0
    for sentence in listWords:
        idx_word = 0
        for word in sentence:
            new_word = []
            idx_charter = 0
            for charter in word:
                idx = 0;
                # numbers
                for i in range(48, 57 + 1):
                    if charter == chr(i):
                        idx = i
                        new_word.append(i)
                # Latin uppers
                for i in range(65, 90 + 1):
                    if charter == chr(i):
                        idx = i
                        new_word.append(i)
                # Latin downs
                for i in range(97, 122 + 1):
                    if charter == chr(i):
                        idx = i
                        new_word.append(i)
                # Cyrillic
                for i in range(1072, 1103 + 1):
                    if charter == chr(i):
                        idx = i
                        new_word.append(i)

                listWordsToNN[idx_sentence][idx_word][idx_charter] = idx
                idx_charter = idx_charter + 1

            idx_word = idx_word + 1
            newListWords.append(new_word)

        idx_sentence = idx_sentence + 1

    # print(newListWords)

    # print(listWordsToNN[0])

    # _________________________________________________________________________________

    # Prepare weights
    # Finding reference words to array words

    # _________________________________________________________________________________

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
    #
    # For Trainging NN

    # _________________________________________________________________________________

    # future_weigths = np.zeros(length_sentence, dtype=float)
    list_future_weigths = np.zeros((len(listWords), count_words), dtype=float)

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
    # print(list_future_weigths)
    # _________________________________________________________________________________

    # Appending feature of applicants to list to json file
    # 1 day for remove from applicants.json
    # 240 it's 50% <- 1 day - 24 hours - 48 query * 10 news
    # 384 it's 80% <- 1 day - 24 hours - 48 query * 10 news
    # 3 day for appending to params.json

    border = 100
    path = 'C:\\Users\\user\\0_Py\\Helper\\Classifier_economics_news\\'

    idx_word = 0
    idx_sentence = 0
    for header in listWords:
        # print(header)
        for obj in header:
            if list_future_weigths[idx_sentence][idx_word] == 0:
                file_name = 'applicants'
                feature_list_applicants = read_data_json(path, file_name)

                # find to feature_list_applicants obj
                success = 0
                # Increase count
                for item in feature_list_applicants:
                    # print(item["name"], item["count"], sep=' ')
                    if obj == item["name"]:
                        item["count"] = item["count"] + 1
                        # print("I found of name! --->>> " + str(item["count"]))
                        file_name = 'applicants'
                        write_data_json(feature_list_applicants, path, file_name)
                        success = 1

                        if item["count"] >= border:
                            rng = np.random.default_rng()
                            file_name = 'params'
                            list_params = read_data_json(path, file_name)

                            list_params.append({"name": item["name"],
                                                "synonyms": [""],
                                                "impact": (rng.random() - 0.5)
                                                })
                            file_name = 'applicants'
                            write_data_json(list_params, path, file_name)
                            feature_list_applicants.remove(item)

                            file_name = 'applicants'
                            write_data_json(feature_list_applicants, path, file_name)

                        break
                # Add new feature
                if success == 0:
                    new_feature_applicant = {"name": obj, "count": 1}
                    feature_list_applicants.append(new_feature_applicant)
                    file_name = 'applicants'
                    write_data_json(feature_list_applicants, path, file_name)
                    # print(obj)

            idx_word = idx_word + 1
        idx_word = 0
        idx_sentence = idx_sentence + 1


    # feature_list_applicants.append()

    # # ______________________________ NN ______________________________

    tickers = ['FXRB',
               'FXMM',
               'FXRU',
               'FXRB',
               'FXWO',
               'FXWR',
               'SU26214RMFS5',
               'RU000A100089',
               'RU000A0ZZH84',
               'RU000A0ZYBS1'
               ]

    # logging.basicConfig(level=logging.DEBUG)

    curr_day = datetime.date(2020, 1, 1)
    # curr_day = datetime.date(datetime.datetime.now().year,
    #                          datetime.datetime.now().month,
    #                          datetime.datetime.now().day)
    # print(curr_day)
    exporter = Exporter()
    data = exporter.lookup(name=tickers[2], market=Market.ETF_MOEX)
    # print(data.head())
    stock = exporter.download(data.index[0], market=Market.ETF_MOEX, start_date=curr_day)
    # print(stock.head())

    file_name = 'stocks_' + str(tickers[2]) + '.csv'
    stock.to_csv(file_name)

    # fxru = pd.read_csv(file_name)

    date_value = stock.get('<DATE>')
    time_value = stock.get('<TIME>')
    open_value = stock.get('<OPEN>')
    close_value = stock.get('<CLOSE>')
    high_value = stock.get('<HIGH>')
    low_value = stock.get('<LOW>')
    volume_value = stock.get('<VOL>')

    # plt.plot(time_value, low_value)
    # close_value.plot()
    # high_value.plot()
    # low_value.plot()
    # volume_value.plot()
    # plt.show()

    list_time_value = time_value.to_list()
    list_open_value = open_value.to_list()
    list_close_value = close_value.to_list()
    list_high_value = high_value.to_list()
    list_low_value = low_value.to_list()
    list_volume_value = volume_value.to_list()

    listOpenValuesToNN = []
    listCloseValuesToNN = []
    listHighValuesToNN = []
    listLowValuesToNN = []
    listVolumeValuesToNN = []
    listTimePointsToNN = []
    for dt_news in time_news:
        for dt in list_time_value:
            regex = r":00$"
            frame_minute = str(dt)
            matches = re.findall(regex, frame_minute)
            frame_minute = frame_minute.replace(matches[0], '')

            if len(frame_minute) < 3:
                frame_minute = frame_minute + ':00'

            if dt_news == frame_minute:
                listTimePointsToNN.append(dt)
                listOpenValuesToNN.append(list_open_value[list_time_value.index(dt)])
                listCloseValuesToNN.append(list_close_value[list_time_value.index(dt)])
                listHighValuesToNN.append(list_high_value[list_time_value.index(dt)])
                listLowValuesToNN.append(list_low_value[list_time_value.index(dt)])
                listVolumeValuesToNN.append(list_volume_value[list_time_value.index(dt)])
                break

            # print(frame_minute)

    # print(listWordsToNN)
    print(listOpenValuesToNN)
    listTrueValue = list_true_value(listOpenValuesToNN)
    print(len(listTrueValue))
    listTrueValue.insert(0, listTrueValue[0])

    # задаем для воспроизводимости результатов
    np.random.seed(2)

    # создаем модели, добавляем слои один за другим
    model = Sequential()
    model.add(Dense(12, input_dim=count_words * count_charters, activation='relu'))  # входной слой требует задать input_dim
    model.add(Dense(15, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # сигмоида вместо relu для определения вероятности

    # компилируем модель, используем градиентный спуск adam
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

    print(len(listTrueValue))
    idx = 0
    for news in listWordsToNN:
        # разбиваем датасет на матрицу параметров (X) и вектор целевой переменной (Y)
        one_sentence_news = news.ravel()
        X = one_sentence_news
        Y = listTrueValue[idx]
        # print(listTrueValue)

        # обучаем нейронную сеть
        model.fit(X, Y, epochs=1000, batch_size=10)

        idx = idx + 1



    # # оцениваем результат
    # scores = model.evaluate(X, Y)
    # print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))


    # # Тренируем нашу нейронную сеть!
    # dim_in = count_words * count_charters
    # dim_h = 10
    # count_h = 5
    # network = OurNeuralNetwork(dim_in, dim_h, count_h)
    # idx = 0
    # for sentence in listWordsToNN:
    #     for word in sentence:
    #         network.train(word, listTrueValue[idx])
    #
    #     idx = idx + 1
    #
    # # # Делаем предсказания
    # # emily = np.array([-7, -3])  # 128 фунтов, 63 дюйма
    # # frank = np.array([20, 2])  # 155 фунтов, 68 дюймов
    # # print("Emily: %.3f" % network.feedforward(emily))  # 0.951 - F
    # # print("Frank: %.3f" % network.feedforward(frank))  # 0.039 - M


if __name__ == '__main__':
    main()
