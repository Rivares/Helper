# coding: UTF-8

import lib_general as my_general
import lib_gui as my_gui


from keras.layers import Dense, Dropout, LSTM, Embedding
from keras.models import Sequential
import keras


start = my_general.datetime.datetime(2020, 1, 1);
end = my_general.datetime.datetime(my_general.datetime.datetime.now().year,
                                   my_general.datetime.datetime.now().month,
                                   my_general.datetime.datetime.now().day);


path_name_class_e_n = 'Classifier_economics_news\\Classifier_e_n.py'
path_name_class_p_n = 'Classifier_politics_news\\Classifier_p_n.py'
path_name_ta_stocks = 'TA_stocks\\TA_stocks.py'
path_name_parser_stocks = 'Parser_market/Parser_market.py'

prediction_e_n = []
prediction_p_n = []
market = []
result_ta = []


def exec_full(file_path, globals=None, locals=None):
    if globals is None:
        globals = {}
    globals.update({
        "__file__": file_path,
        "__name__": "__main__",
    })
    with open(file_path, 'rb') as file:
        exec(compile(file.read(), file_path, 'exec'), globals, locals)


def main():
    # app = my_gui.MainApp()
    # app.run()

    my_general.name_ticker = 'FXRB ETF'
    my_general.root_path = 'C:\\Users\\user\\0_Py\\'
    root_path = 'C:\\Users\\user\\0_Py\\'

    while (my_general.datetime.datetime.now().hour > 9) and (my_general.datetime.datetime.now().hour < 23):

        exec_full(path_name_class_e_n)
        exec_full(path_name_class_p_n)
        exec_full(path_name_ta_stocks)
        exec_full(path_name_parser_stocks)

        path = 'Helper\\Classifier_economics_news\\'
        filename = 'prediction_e_n'
        prediction_e_n = my_general.read_data_json(root_path + path, filename)

        path = 'Helper\\Classifier_politics_news\\'
        filename = 'prediction_p_n'
        prediction_p_n = my_general.read_data_json(root_path + path, filename)

        path = 'Helper\\TA_stocks\\'
        filename = 'result_ta'
        result_ta = my_general.read_data_json(root_path + path, filename)

        path = 'Helper\\Parser_market\\'
        filename = 'market'
        market = my_general.read_data_json(root_path + path, filename)

        # print(prediction_e_n)
        # print(prediction_p_n)
        # print(market)
        # print(result_ta)

        print("__________________ Global training __________________")

        my_general.np.random.seed(2)
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

        input_nodes = my_general.np.asarray(input_nodes, dtype=my_general.np.float32)
        output_nodes = my_general.np.asarray(output_nodes, dtype=my_general.np.float32)

        input_nodes = input_nodes.reshape((1, 1, count_inputs))
        output_nodes = output_nodes.reshape((1, 1))
        # print(input_nodes.shape)
        # print(output_nodes.shape)

        path = root_path + 'Helper\\'
        filename = 'X'
        my_general.write_data_json(X, path, filename)

        filename = 'Y'
        my_general.write_data_json(Y, path, filename)

        # print(output_nodes)

        if my_general.os.path.exists(model_name) != False:
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
        my_general.write_data_json(main_prediction, path, file_name_prediction)

        # except:
        #     print("Problem with – fit(Global)!")

    else:
        print("Sleep...")


if __name__ == '__main__':
    main()
