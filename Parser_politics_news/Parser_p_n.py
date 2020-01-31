from openpyxl import Workbook
from bs4 import BeautifulSoup
import datetime
import requests
import csv

file_name = 'politics_news'
extension = '.csv'

def get_html(url):
    r = requests.get(url)
    return r.text

def convert_csv_to_xls():
    wb = Workbook()
    ws = wb.active
    with open(file_name + extension, 'r') as f:
        for row in csv.reader(f):
            ws.append(row)
    wb.save(file_name + '.xlsx')


def write_csv(data):
    with open(file_name + extension, 'a', newline='') as f:
        fieldnames = ['title', 'additionally', 'href', 'date']
        writer = csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'title' : data['title'],
                         'additionally': data['additionally'],
                         'href' : data['href'],
                         'date': data['date']
                         })


def get_page_data(html):
    soup = BeautifulSoup(html, 'lxml')
    divs = soup.find('div', class_='list list-tags')
    ads = divs.find_all('div', class_='list-item', limit = 10)

    title = ''
    addit= ''
    href = ''
    date = ''
    for ad in ads:
        try:
            div = ad.find('div', class_='list-item__content').find('a', class_='list-item__title color-font-hover-only');
            title = div.text;

            href = div.get('href');

            title_picture = ad.find('div', class_ = 'list-item__content')\
                              .find('a', class_ = 'list-item__image')\
                              .find('picture')\
                              .find('img')\
                              .get('title')

            addit = title_picture;

            now = datetime.datetime.now()

            date = str(now.year) + '/' + str(now.month) + '/' + str(now.day);
            data = {'title': title, 'additionally': addit, 'href': href, 'date': date}


            write_csv(data)
            convert_csv_to_xls()
        except:
                title = 'Error'
                addit = 'Error'
                href = 'Error'
                date = 'Error'



def main():
    base_url = "https://ria.ru/politics/"

    for i in range(1, 11):       # 8 is maximum
        url_gen = base_url
        html = get_html(url_gen)
        get_page_data(html)


if __name__ == '__main__':
    main()
