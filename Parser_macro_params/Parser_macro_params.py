from bs4 import BeautifulSoup
from openpyxl import Workbook
import matplotlib.pyplot as plt
import pandas as pd
import requests
import datetime
import logging
import json
import re


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
    tr = soup.find('table', class_='simple-little-table little trades-table').find_all('tr')
    # print(tr)

    data.clear()
    list_indicators = []

    for item in tr:
        item_indic = []
        tds = item.find_all('td')

        for td in tds:
            if td is not '':
                td = td.text
                item_indic.append(td)
                # print(td)

        if len(item_indic) > 1:
            list_indicators.append(item_indic)

        # print(item_indic)

    # print(len(list_indicators))
    # print(list_indicators)

    return list_indicators


def main():
    base_url = "https://smart-lab.ru/q/shares_fundamental/"
    list_micro_data = [[]]

    # os.remove(file_name + '.json')

    url_gen = base_url
    html = get_html(url_gen)
    list_micro_data = get_page_data(html, list_micro_data)

    list_micro_data.pop(0)
    for ticker in list_micro_data:
        ticker.pop(0)               # Delete №
        ticker.pop(1)               # Delete Name
        ticker.pop(-1)              # Delete report name
        for idx in range(0, len(ticker)):
            if ticker[idx] == '':
                ticker[idx] = 0

        # print(ticker)
        # print(len(ticker))

    reg = re.compile('[aA-zZа-яА-Я+%]')

    for ticker in list_micro_data:
        for idx in range(0, len(ticker)):
            ticker[idx] = str(ticker[idx]).lower()
            ticker[idx] = reg.sub('', ticker[idx])

        print(ticker)


if __name__ == '__main__':
    main()
