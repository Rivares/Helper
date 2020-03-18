# coding: UTF-8

import lib_general as my_general


root_path = my_general.root_path
curr_ticker = my_general.name_ticker


spider = \
{
    'title': '',
    'additionally': '',
    'href': '',
    'time': 0
}

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


def write_article_csv(file_name, data):
    with open(file_name + '.csv', 'a', newline='') as f:
        fieldnames = ['title', 'additionally', 'href', 'time']
        writer = my_general.csv.DictWriter(f, delimiter=',', fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'title': data['title'],
                         'additionally': data['additionally'],
                         'href': data['href'],
                         'time': data['time']
                         })


def read_article_csv():
    path = root_path + 'Helper\\Classifier_politics_news\\'
    file_name = 'politics_news'
    extension = '.csv'
    listSpider_E_N = []

    if my_general.os.stat(path + file_name + extension).st_size != 0:
        with open(path + file_name + extension, newline='\n') as csvfile:
            reader = my_general.csv.DictReader(csvfile, delimiter=',')
            for row in reader:
                # reader.fieldnames[i] - i = 0 - title; 1 - additionally; 2 - href; 3 - time
                if str(row.get(reader.fieldnames[0])) != str(reader.fieldnames[0]):
                    listSpider_E_N.append(
                        spider(row.pop(reader.fieldnames[0])
                               , row.pop(reader.fieldnames[1])
                               , row.pop(reader.fieldnames[2])
                               , row.pop(reader.fieldnames[3])
                               )
                    )
    else:
        print("Error read file!")
    return listSpider_E_N


def read_params_xlsx():
    country_path = root_path + 'Helper\\Classifier_politics_news\\'
    country_file_name = 'params'
    country_extension = '.xlsx'

    workbook = my_general.xlrd.open_workbook(country_path + country_file_name + country_extension, on_demand=True)
    worksheet = workbook.sheet_by_index(0)

    if my_general.os.stat(country_path + country_file_name + country_extension).st_size != 0:
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


def main():
    return


if __name__ == '__main__':
    main()
