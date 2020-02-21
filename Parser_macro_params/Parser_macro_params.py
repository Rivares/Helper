from bs4 import BeautifulSoup
from openpyxl import Workbook
import matplotlib.pyplot as plt
import pandas as pd
import requests
import datetime
import logging
import json
import csv

start = datetime.datetime(2020, 1, 1);
end = datetime.datetime(datetime.datetime.now().year,
                        datetime.datetime.now().month,
                        datetime.datetime.now().day)
file_name = 'macro_params'
extansion = '.json'


# ______________________________ Parser ______________________________

class Spider(object):
    def __init__(self, header, unit, value, wordify, year):
        """Constructor"""
        self.header = header
        self.unit = unit
        self.value = value
        self.wordify = wordify
        self.year = year

    def get_params(self):
        return self.header, self.unit, self.value, self.wordify, self.year


def get_html(url):
    r = requests.get(url)
    return r.text


def get_page_data(html, data):
    soup = BeautifulSoup(html, 'lxml')
    imfs = soup.find('div', class_='dm-wrapper').find('div', class_='dm-dataset-panels')
    print(imfs)
    divs = imfs.find('imf-dataset-panel', class_='default')
    # profile-page > div > div.dm-dataset-panels > imf-dataset-panel.default.expanded > div
    # document.querySelector("#profile-page > div > div.dm-dataset-panels > imf-dataset-panel.default.expanded > div")
    # /html/body/imf-component/div/div[3]/imf-dataset-panel[1]/div

    print(divs)

    data.clear()
    header = ''
    unit = ''
    value = ''
    wordify = ''
    year = ''
    # for ad in ads:
    #     try:
    #         div = ad.find('div', class_='list-item__content').find('a',
    #                                                                class_='list-item__title color-font-hover-only');
    #         title = div.text;
    #
    #         href = div.get('href');
    #
    #         title_picture = ad.find('div', class_='list-item__content') \
    #             .find('a', class_='list-item__image') \
    #             .find('picture') \
    #             .find('img') \
    #             .get('title')
    #         addit = title_picture;
    #
    #         time = ad.find('div', class_='list-item__info') \
    #             .find('div', class_='list-item__date').text
    #         time = time.split(', ')
    #         time = time[-1]
    #
    #         data.append({'header': header, 'unit': unit, 'value': value, 'wordify': wordify, 'year': year})
    #
    #     except:
    #         header = 'Error'
    #         unit = 'Error'
    #         value = 'Error'
    #         wordify = 'Error'
    #         year = 'Error'

    return data


def main():
    base_url = "https://www.imf.org/external/datamapper/profile/RUS/WEO"
    macro_data = []

    # os.remove(file_name + '.json')

    url_gen = base_url
    html = get_html(url_gen)
    article_data = get_page_data(html, macro_data)

    print("Parsed data ---->>> ")
    print(article_data)


if __name__ == '__main__':
    main()
