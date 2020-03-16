# coding: UTF-8

import lib_general as my_lib

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.app import App


from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.models import Sequential
import keras


red = [1, 0, 0, 1]
green = [0, 1, 0, 1]
blue = [0, 0, 1, 1]
purple = [1, 0, 1, 1]


class HBoxLayoutExample(App):
    def build(self):
        layout = BoxLayout(padding=10)
        colors = [red, green, blue, purple]

        button = Button(text='Hello from Kivy',
                        background_color=green)
        button.bind(on_press=self.on_press_button)
        layout.add_widget(button)
        for i in range(5):
            btn = Button(text="Button #%s" % (i + 1),
                         background_color=red
                         )

            layout.add_widget(btn)
        return layout

    def on_press_button(self, instance):
        print('Вы нажали на кнопку!')


root_path = 'C:\\Users\\user\\0_Py\\'

SYMBOLS = ['FXRB',
           'FXMM',
           'FXRU',
           'FXRB',
           'FXWO',
           'FXWR'
           ]

start = my_lib.datetime.datetime(2020, 1, 1);
end = my_lib.datetime.datetime(my_lib.datetime.datetime.now().year,
                               my_lib.datetime.datetime.now().month,
                               my_lib.datetime.datetime.now().day);


path_name_class_e_n = 'Classifier_economics_news\\Classifier_e_n.py'
path_name_class_p_n = 'Classifier_politics_news\\Classifier_p_n.py'
path_name_ta_stocks = 'TA_stocks\\TA_stocks.py'
path_name_parser_stocks = 'Parser_market/Parser_market.py'

prediction_e_n = []
prediction_p_n = []
market = []
result_ta = []


def exec_full(file_path):
    global_namespace = {
        "__file__": file_path,
        "__name__": "__main__",
    }
    with open(file_path, 'rb') as file:
        exec(compile(file.read(), file_path, 'exec'), global_namespace)


def main():
    # app = HBoxLayoutExample()
    # app.run()

    while (my_lib.datetime.datetime.now().hour > 9) and (my_lib.datetime.datetime.now().hour < 23):

        exec_full(path_name_class_e_n)
        exec_full(path_name_class_p_n)
        exec_full(path_name_ta_stocks)
        exec_full(path_name_parser_stocks)

        path = 'Helper\\Classifier_economics_news\\'
        filename = 'prediction_e_n'
        prediction_e_n = my_lib.read_data_json(root_path + path, filename)

        path = 'Helper\\Classifier_politics_news\\'
        filename = 'prediction_p_n'
        prediction_p_n = my_lib.read_data_json(root_path + path, filename)

        path = 'Helper\\TA_stocks\\'
        filename = 'result_ta'
        result_ta = my_lib.read_data_json(root_path + path, filename)

        path = 'Helper\\Parser_market\\'
        filename = 'market'
        market = my_lib.read_data_json(root_path + path, filename)

        # print(prediction_e_n)
        # print(prediction_p_n)
        # print(market)
        # print(result_ta)

        print("__________________ Global training __________________")

        my_lib.np.random.seed(2)
        path = 'Helper\\'
        model_name = root_path + path + 'NN_Main_model.h5'

        X = []
        Y = []

        Y.append(result_ta[0]['diff_value'])

        X.append(prediction_e_n['score'])

        X.append(prediction_p_n['score'])

        for ticker in market:
            for input in ticker:
                X.append(input['open_value'])
                X.append(input['close_value'])
                X.append(input['high_value'])
                X.append(input['low_value'])
                X.append(input['volume_value'])

        X.append(result_ta[0]['open_value'])
        X.append(result_ta[0]['close_value'])
        X.append(result_ta[0]['high_value'])
        X.append(result_ta[0]['low_value'])
        X.append(result_ta[0]['volume_value'])
        X.append(result_ta[0]['adi_i'])
        X.append(result_ta[0]['adx_aver'])
        X.append(result_ta[0]['adx_DI_pos'])
        X.append(result_ta[0]['adx_DI_neg'])
        X.append(result_ta[0]['ai_i'])
        X.append(result_ta[0]['ai_up'])
        X.append(result_ta[0]['ai_down'])
        X.append(result_ta[0]['ao_i'])
        X.append(result_ta[0]['atr_i'])
        X.append(result_ta[0]['bb_bbh'])
        X.append(result_ta[0]['bb_bbl'])
        X.append(result_ta[0]['bb_bbm'])
        X.append(result_ta[0]['ccl_i'])
        X.append(result_ta[0]['cmf_i'])
        X.append(result_ta[0]['cmf_signal'])
        X.append(result_ta[0]['cr_i'])

        X.append(result_ta[0]['dc_dch'])
        X.append(result_ta[0]['dc_dcl'])
        X.append(result_ta[0]['dlr_i'])
        X.append(result_ta[0]['dpo_i'])
        X.append(result_ta[0]['ema_i'])
        X.append(result_ta[0]['fi_i'])
        X.append(result_ta[0]['ichimoku_a'])
        X.append(result_ta[0]['ichimoku_b'])
        X.append(result_ta[0]['kama_i'])
        X.append(result_ta[0]['kc_kcc'])
        X.append(result_ta[0]['kc_kch'])
        X.append(result_ta[0]['kc_kcl'])
        X.append(result_ta[0]['kst'])
        X.append(result_ta[0]['kst_diff'])
        X.append(result_ta[0]['kst_sig'])
        X.append(result_ta[0]['vi_diff'])
        X.append(result_ta[0]['vi_neg'])
        X.append(result_ta[0]['vi_pos'])

        X.append(result_ta[0]['mfi_i'])
        X.append(result_ta[0]['mi'])
        X.append(result_ta[0]['nvi_i'])
        X.append(result_ta[0]['obv_i'])
        X.append(result_ta[0]['psar_i'])
        X.append(result_ta[0]['psar_up'])
        X.append(result_ta[0]['psar_down'])
        X.append(result_ta[0]['roc_i'])
        X.append(result_ta[0]['rsi_i'])
        X.append(result_ta[0]['stoch_i'])
        X.append(result_ta[0]['stoch_signal'])
        X.append(result_ta[0]['trix_i'])
        X.append(result_ta[0]['tsi_i'])
        X.append(result_ta[0]['uo_i'])
        X.append(result_ta[0]['vpt_i'])

        count_inputs = len(X)
        print("Len NN: " + str(count_inputs))
        print("X: "); print(X)
        print("Y: "); print(Y)

        # создаем модели, добавляем слои один за другим
        model = Sequential()

        model.add(LSTM(int(count_inputs / 2), return_sequences=True, input_shape=(1, count_inputs)))
        model.add(LSTM(int(count_inputs / 4), return_sequences=True))
        model.add(LSTM(int(count_inputs / 6), return_sequences=True))
        model.add(LSTM(int(count_inputs / 8)))
        model.add(Dense(int(count_inputs / 10), activation='relu'))
        model.add(Dense(int(count_inputs / 12), activation='relu'))
        model.add(Dense(int(count_inputs / 14), activation='softmax'))
        model.add(Dense(int(count_inputs / 16), activation='softmax'))
        model.add(Dense(int(count_inputs / 18), activation='tanh'))
        model.add(Dense(int(count_inputs / 20), activation='tanh'))
        model.add(Dense(int(count_inputs / 40), activation='sigmoid'))
        model.add(Dense(int(count_inputs / 60), activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid'))

        # model.summary()

        model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])

        input_nodes = []
        output_nodes = []
        input_nodes.append(X)
        output_nodes.append(Y)

        input_nodes = my_lib.np.asarray(input_nodes, dtype=my_lib.np.float32)
        output_nodes = my_lib.np.asarray(output_nodes, dtype=my_lib.np.float32)

        input_nodes = input_nodes.reshape((1, 1, count_inputs))
        output_nodes = output_nodes.reshape((1, 1))
        # print(input_nodes.shape)
        # print(output_nodes.shape)

        path = root_path + 'Helper\\'
        filename = 'X'
        my_lib.write_data_json(X, path, filename)

        filename = 'Y'
        my_lib.write_data_json(Y, path, filename)

        # print(output_nodes)

        if my_lib.os.path.exists(model_name) != False:
            # Recreate the exact same model
            new_model = keras.models.load_model(model_name)
        else:
            new_model = model

        # try:
        # обучаем нейронную сеть
        history = new_model.fit(input_nodes, output_nodes, epochs=1, batch_size=64)

        # Export the model to a SavedModel
        new_model.save(model_name)

        # оцениваем результат
        scores = new_model.predict(input_nodes)

        main_prediction = {"score": float(scores[-1] * 100)}
        print(main_prediction)

        path = root_path + 'Helper\\'
        file_name_prediction = 'main_prediction'
        my_lib.write_data_json(main_prediction, path, file_name_prediction)

        # except:
        #     print("Problem with – fit(Global)!")

    else:
        print("Sleep...")


if __name__ == '__main__':
    main()
