# coding: UTF-8

import lib_general as my_lib

from keras.models import Sequential
from keras.layers import Dense, Dropout
import keras


root_path = 'C:\\Users\\user\\0_Py\\'


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
        writer = my_lib.csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'title': data['title'],
                         'additionally': data['additionally'],
                         'href': data['href'],
                         'time': data['time']
                         })


def read_article_csv():
    path = root_path + 'Helper\\Classifier_economics_news\\'
    file_name = 'economics_news'
    extension = '.csv'
    listSpider_E_N = []

    if my_lib.os.stat(path + file_name + extension).st_size != 0:
        with open(path + file_name + extension, newline='\n') as csvfile:
            reader = my_lib.csv.DictReader(csvfile, delimiter=',')
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


def read_params_xlsx():
    country_path = root_path + 'Helper\\Classifier_economics_news\\'
    country_file_name = 'params'
    country_extension = '.xlsx'

    workbook = my_lib.xlrd.open_workbook(country_path + country_file_name + country_extension, on_demand=True)
    worksheet = workbook.sheet_by_index(0)

    if my_lib.os.stat(country_path + country_file_name + country_extension).st_size != 0:
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


# ______________________________ Parser ______________________________


def get_html(url):
    r = my_lib.requests.get(url)
    return r.text


def get_page_data(html, article_data):
    soup = my_lib.BeautifulSoup(html, 'lxml')
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


def main():
    print("\n__________________ Economic news __________________\n")

    base_url = "https://ria.ru/economy/"
    article_data = []
    hash_news_e_n = []

    # os.remove(file_name + '.csv')
    # os.remove(file_name + '.xlsx')
    # os.remove(file_name + '.json')

    url_gen = base_url
    html = get_html(url_gen)
    article_data = get_page_data(html, article_data)

    # print(article_data.__len__())
    path = root_path + 'Helper\\Classifier_economics_news\\'
    file_name = 'economics_news'
    my_lib.write_data_json(article_data, path, file_name)

    # _________________________________________________________________________________

    # Check on repeat

    hash_news_e_n = my_lib.read_data_json(path, 'hash_news_e_n')

    file_name = 'economics_news'
    if my_lib.md5(path + file_name + '.json') == hash_news_e_n[0]["hash"]:
        print("___ No the new economics news ___")
        return

    # _________________________________________________________________________________

    count_sentences = article_data.__len__()
    count_words = 30
    count_charters = 30

    # _________________________________________________________________________________

    # Creating list of news + to Lower Case + delete ',' and  '.'

    path = root_path + 'Helper\\Classifier_economics_news\\'
    file_name = 'economics_news'
    news = my_lib.read_data_json(path, file_name)

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

    reg = my_lib.re.compile('[^а-яА-Я -]')

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

    morph = my_lib.pymorphy2.MorphAnalyzer()

    for obj in listSpider_E_N:
        obj.title = (' '.join([morph.normal_forms(w)[0] for w in obj.title.split()]))
        obj.additionally = (' '.join([morph.normal_forms(w)[0] for w in obj.additionally.split()]))

    # _________________________________________________________________________________

    # Read reference words from json file

    # listParams_E_N = read_params_xlsx()
    path = root_path + 'Helper\\Classifier_economics_news\\'
    file_name = 'params'
    listParams_E_N = my_lib.read_data_json(path, file_name)
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
            if (p.tag.POS == 'ADVB') or \
                    (p.tag.POS == 'NPRO') or \
                    (p.tag.POS == 'PRED') or \
                    (p.tag.POS == 'PREP') or \
                    (p.tag.POS == 'CONJ') or \
                    (p.tag.POS == 'PRCL') or \
                    (p.tag.POS == 'INTJ'):
                sentence.remove(word)
        # print(sentence)
    # _________________________________________________________________________________

    # Transform to digital mode

    # print(listWords[0][0])

    newListWords = []
    listWordsToNN = my_lib.np.zeros((count_sentences, count_words, count_charters))

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

    # For Trainging NN

    # _________________________________________________________________________________

    # future_weigths = np.zeros(length_sentence, dtype=float)
    list_future_weigths = my_lib.np.zeros((len(listWords), count_words), dtype=float)

    idx_word = 0
    idx_sentence = 0
    for header in listWords:
        # print(header)
        for obj in header:
            # print(obj.lower())
            for params in listParams_E_N:
                if my_lib.fuzz.ratio(params.get('name'), obj.lower()) > 90:
                    # print("I found of name! --->>> " + str(obj))
                    list_future_weigths[idx_sentence][idx_word] = float(params.get('impact'))
                    break
                else:
                    if len(params.get('synonyms')) >= 1:
                        for it in params.get('synonyms'):
                            if my_lib.fuzz.ratio(str(it), str(obj.lower())) > 80:
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
    path = root_path + 'Helper\\Classifier_economics_news\\'

    idx_word = 0
    idx_sentence = 0
    for header in listWords:
        # print(header)
        for obj in header:
            if list_future_weigths[idx_sentence][idx_word] == 0:
                file_name = 'applicants'
                feature_list_applicants = my_lib.read_data_json(path, file_name)

                # find to feature_list_applicants obj
                success = 0
                # Increase count
                for item in feature_list_applicants:
                    # print(item["name"], item["count"], sep=' ')
                    if obj == item["name"]:
                        item["count"] = item["count"] + 1
                        # print("I found of name! --->>> " + str(item["count"]))
                        file_name = 'applicants'
                        my_lib.write_data_json(feature_list_applicants, path, file_name)
                        success = 1

                        if item["count"] >= border:
                            rng = my_lib.np.random.default_rng()
                            file_name = 'params'
                            list_params = my_lib.read_data_json(path, file_name)

                            list_params.append({"name": item["name"],
                                                "synonyms": [""],
                                                "impact": (rng.random() - 0.5)
                                                })
                            file_name = 'params'
                            my_lib.write_data_json(list_params, path, file_name)
                            feature_list_applicants.remove(item)

                            file_name = 'applicants'
                            my_lib.write_data_json(feature_list_applicants, path, file_name)

                        break
                # Add new feature
                if success == 0:
                    new_feature_applicant = {"name": obj, "count": 1}
                    feature_list_applicants.append(new_feature_applicant)
                    file_name = 'applicants'
                    my_lib.write_data_json(feature_list_applicants, path, file_name)
                    # print(obj)

            idx_word = idx_word + 1
        idx_word = 0
        idx_sentence = idx_sentence + 1


    # feature_list_applicants.append()

    # ______________________________ NN ______________________________

    tickers = ['FXRB',
               'FXMM',
               'FXRU',
               'FXRB',
               'FXWO',
               'FXWR'
               ]

    # logging.basicConfig(level=logging.DEBUG)

    # curr_day = datetime.date(2020, 1, 1)
    curr_day = my_lib.datetime.date(my_lib.datetime.datetime.now().year,
                                    my_lib.datetime.datetime.now().month,
                                    my_lib.datetime.datetime.now().day)
    # print(curr_day)
    exporter = my_lib.Exporter()
    data = exporter.lookup(name=tickers[0], market=my_lib.Market.ETF_MOEX)
    # print(data.head())
    stock = exporter.download(data.index[0], market=my_lib.Market.ETF_MOEX, start_date=curr_day)
    # print(stock.head())

    file_name = path + 'stocks_' + str(tickers[0]) + '.csv'
    stock.to_csv(file_name)

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
    list_distances = []
    list_distances.append(0)

    for dt_news in time_news:
        for dt in list_time_value:
            regex = r":00$"
            frame_minute = str(dt)
            matches = my_lib.re.findall(regex, frame_minute)
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

    listOpenValuesToNN.reverse()
    listCloseValuesToNN.reverse()
    listHighValuesToNN.reverse()
    listLowValuesToNN.reverse()
    listVolumeValuesToNN.reverse()
    listTimePointsToNN.reverse()

    time_point = list_time_value[0]
    listOpenValuesToNN.insert(0, list_open_value[list_time_value.index(time_point)])
    listCloseValuesToNN.insert(0, list_open_value[list_time_value.index(time_point)])
    listHighValuesToNN.insert(0, list_open_value[list_time_value.index(time_point)])
    listLowValuesToNN.insert(0, list_open_value[list_time_value.index(time_point)])
    listVolumeValuesToNN.insert(0, list_open_value[list_time_value.index(time_point)])
    listTimePointsToNN.insert(0, time_point)

    # print(listWordsToNN)
    # print(listOpenValuesToNN)

    if len(listOpenValuesToNN) > 0:
        # Morning
        if len(listOpenValuesToNN) < 10:
            size = 10 - len(listOpenValuesToNN)
            firstValue = listOpenValuesToNN[0]
            for item in range(0, size):
                listOpenValuesToNN.insert(0, firstValue)

            for idx in range(0, len(listTimePointsToNN) - 1):
                curr_i = str(listTimePointsToNN[idx])
                next_i = str(listTimePointsToNN[idx + 1])
                list_distances.append(int(next_i.replace(':', '')) - int(curr_i.replace(':', '')))

            # print(sum(list_distances))

        # print(listOpenValuesToNN)
        # print(len(listOpenValuesToNN))

        listOpenValuesToNN.insert(0, listOpenValuesToNN[0])
        listCloseValuesToNN.insert(0, listCloseValuesToNN[0])
        listHighValuesToNN.insert(0, listHighValuesToNN[0])
        listLowValuesToNN.insert(0, listLowValuesToNN[0])
        listVolumeValuesToNN.insert(0, listVolumeValuesToNN[0])
        listTimePointsToNN.insert(0, listTimePointsToNN[0])

        listTrueValue = my_lib.list_true_value(listOpenValuesToNN)
        # print(listTrueValue)
        # print(len(listTrueValue))

        # задаем для воспроизводимости результатов
        my_lib.np.random.seed(2)
        model_name = path + 'NN_model.h5'

        # создаем модели, добавляем слои один за другим
        model = Sequential()
        model.add(Dense(5 * count_words, input_dim=(count_words * count_charters), activation='relu'))  # 0
        model.add(Dense(4 * count_words, activation='relu'))    # 1
        model.add(Dense(3 * count_words, activation='tanh'))    # 2
        model.add(Dense(2 * count_words, activation='tanh'))    # 3
        model.add(Dense(count_words, activation='tanh'))        # 4
        model.add(Dense(count_words - 10, activation='sigmoid'))
        model.add(Dropout(0.2))
        model.add(Dense(count_words - 20, activation='sigmoid'))
        model.add(Dropout(0.2))
        model.add(Dense(count_words - 25, activation='sigmoid'))
        model.add(Dense(count_words - 27, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))

        number_layer_words = 5
        native_weights = model.layers[number_layer_words].get_weights()[0]  # 0 - weights
        native_biases = model.layers[number_layer_words].get_weights()[1]   # 1 - biases

        # print("Old")
        # print(len(native_weights))

        new_weights = my_lib.np.zeros((len(native_weights), len(native_weights[0])), dtype=float)
        for future_news in list_future_weigths:
            idx_1 = 0
            for weights in native_weights:
                add = future_news[idx_1]

                idx_2 = 0
                for weight in weights:
                    new_weights[idx_1][idx_2] = float(weight + add)
                    idx_2 = idx_2 + 1

                idx_1 = idx_1 + 1

            # print("New")
            # print(len(new_weights))
            keras_weights = [new_weights, native_biases]
            model.layers[number_layer_words].set_weights(keras_weights)

            # компилируем модель, используем градиентный спуск adam
            model.compile(loss="mean_squared_error", optimizer="rmsprop", metrics=['accuracy'])

            X = []

            for news in listWordsToNN:
                # разбиваем датасет на матрицу параметров (X) и вектор целевой переменной (Y)
                one_sentence_news = news.ravel()

                X.append(one_sentence_news)

            X = my_lib.np.asarray(X, dtype=my_lib.np.float32)
            Y = my_lib.np.asarray(listTrueValue, dtype=my_lib.np.float32)

            if my_lib.os.path.exists(model_name) != False:
                # Recreate the exact same model
                new_model = keras.models.load_model(model_name)
            else:
                new_model = model

            try:
                # обучаем нейронную сеть
                history = new_model.fit(X, Y, epochs=1, batch_size=64)

                # Export the model to a SavedModel
                new_model.save(model_name)

                # # evaluate the model
                # scores = model.evaluate(X, Y)
                # print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

                # оцениваем результат
                scores = new_model.predict(X)
                print("\n%s: %.2f%%" % (new_model.metrics_names[1], scores[1] * 100))

                print(scores)
                prediction = {"score": float(scores[-1])}
                print(prediction)

                path = root_path + 'Helper\\Classifier_economics_news\\'
                file_name_prediction = 'prediction_e_n'

                my_lib.write_data_json(prediction, path, file_name_prediction)

            except:
                print("Problem with – fit(C_E_N)!")

    path = root_path + 'Helper\\Classifier_economics_news\\'
    hash_news_e_n = [{"hash": my_lib.md5(path + 'economics_news' + '.json')}]

    file_name = 'hash_news_e_n'
    my_lib.write_data_json(hash_news_e_n, path, file_name)


if __name__ == '__main__':
    main()
