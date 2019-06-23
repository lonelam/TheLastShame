# ./network.py
import keras
from util.constants import *
import tensorflow as tf
import numpy as np
import os
import csv
import time
from sklearn.decomposition import PCA
import random
import pickle
from util.ploter import plot_from_point_list, Framework, get_frame_shape
from residual_network import get_compiled_residual_model
import matplotlib.pyplot as plt
import pyprind
from sklearn.preprocessing import normalize

def parse_ansys_export(file):
    data = []
    with open(file, 'r') as f:
        reader = csv.reader(f,)
        for line in reader:
            try:
                data.append([float(line[3]), ])
            except ValueError:
                continue
            except IndexError:
                continue
    return data


def get_raw_data(need_flatten = True, max_file_read = 123456789):
    raw_data = []
    read_count = 0
    for f in pyprind.prog_bar(list(f for f in os.listdir(FILE_PATH) if f.startswith(DEFAULT_FILE_PREFIX)), stream=1, title='reading raw data'):
        raw_data_row = parse_ansys_export(os.path.join(FILE_PATH, f))
        if need_flatten:
            raw_data_row = np.reshape(raw_data_row, (-1, ))
        if read_count % M_GEN == 0:
            raw_data.append(raw_data_row)
        read_count += 1
        if read_count == max_file_read:
            break
    return raw_data


def get_data(n_seq = 10, need_flatten = True, max_file_read = 123456789, generate_from=0):
    cache_fname = "data_{0}_{1}_{2}_{3}_{4}_{5}.pickle".format(n_seq, need_flatten, max_file_read, DEFAULT_FILE_PREFIX, generate_from, M_GEN)
    try:
        with open(cache_fname, 'rb') as f:
            ret = pickle.load(f)
            print("read data from cache.")
            return ret
    except FileNotFoundError:
        pass
    except EOFError:
        os.remove(cache_fname)

    raw_data = get_raw_data(need_flatten, max_file_read)
    n = len(raw_data)
    if n < n_seq:
        raise FileNotFoundError("there should be more data")
    data = []
    for i in range(generate_from, n - n_seq):
        data_row = []
        for j in range(n_seq):
            data_row.append(raw_data[j + i])
        data.append(data_row)

    #shufle data
    # np.random.shuffle(data)
    result = (np.array(data), raw_data)
    try:
        with open(cache_fname, 'wb') as f:
            pickle.dump(result, f)
    except OverflowError:
        os.remove(cache_fname)
        print("overflow error")
    except MemoryError:
        print("memory error")
        pass
    return result

def predict_internal_data(model, raw_data, n_seq = 10):
    n = len(raw_data)
    if n < n_seq:
        raise FileNotFoundError("there should be more data")
    data = []
    last_prediction = None
    for i in range(0, n - n_seq):
        data_row = []
        for j in range(n_seq):
            if j == n_seq - 2:
                if last_prediction is not None:
                    data_row.append(last_prediction)
                else:
                    data_row.append(raw_data[j + i])
                last_prediction = np.reshape(model.predict(np.array((raw_data[i:i+n_seq - 1],))), (-1,))
            else:
                data_row.append(raw_data[j+i])
        data.append(np.array(data_row))
    return data

def get_compiled_model(input_shape, width=128):
    layers = [
        # keras.layers.Reshape((-1, 1), input_shape=input_shape),
        keras.layers.LSTM(width, input_shape=input_shape, return_sequences=True, dropout=0.1, activation='tanh'),
        keras.layers.LSTM(width, input_shape=(input_shape[0], width), return_sequences=True, dropout=0.1, activation='tanh'),
        keras.layers.LSTM(width, input_shape=(input_shape[0], width), return_sequences=False, dropout=0.1, activation='tanh'),
        keras.layers.Dense(input_shape[1])
    ]

    model = keras.Sequential(layers)
    model.compile(optimizer="adam", loss='mse', metrics=['mae'])
    model.summary()
    return model

class MyPca():
    def __init__(self, n_components):
        self.n_components = n_components
    def fit(self, data):
        self.clip = []
        data = np.array(data, copy=True)
        # normalization
        for i in range(N_FEATURE - 1):
            self.clip.append(self.n_components // N_FEATURE)
        self.clip.append(self.n_components - self.n_components // N_FEATURE * (N_FEATURE - 1))
        data = np.reshape(data, (data.shape[0], -1, N_FEATURE))
        # self.min_list = list(map(np.min, [data[:, :, i] for i in range(N_FEATURE)]))
        # self.max_list = list(map(np.max, [data[:, :, i] for i in range(N_FEATURE)]))
        # self.scale_list = [ self.max_list[i] - self.min_list[i] for i in range(N_FEATURE)]
        # for i in pyprind.prog_bar(range(data.shape[0]), title="Pca fitting."):
        #     for j in range(data.shape[1]):
        #         for k in range(data.shape[2]):
        #             data[i][j][k] = (data[i][j][k] - self.min_list[k]) / self.scale_list[k]
        self.pcas = []
        self.afterPODscale_list = []
        for i in range(N_FEATURE):
            pcai = PCA(self.clip[i])
            transformed = pcai.fit_transform(data[:,:,i])
            print(pcai.explained_variance_ratio_)
            self.afterPODscale_list.append(pcai.explained_variance_ratio_)
            plt.plot(pcai.explained_variance_ratio_)
            plt.show()
            self.pcas.append(pcai)
        # self.afterPODscale_list = np.stack(self.afterPODscale_list, axis=0)

    def transform(self, X):
        X = np.copy(np.reshape(X, (len(X), -1, N_FEATURE)))
        # for i in range(X.shape[0]):
        #     for j in range(X.shape[1]):
        #         for k in range(X.shape[2]):
        #             X[i][j][k] = (X[i][j][k] - self.min_list[k]) / self.scale_list[k]
        transformed_X = []
        for i in range(N_FEATURE):
            transformed_row = self.pcas[i].transform(X[:, :, i])
            for j in range(self.clip[i]):
                transformed_row[:, j] = transformed_row[:, j] * self.afterPODscale_list[i][j]
            transformed_X.append(transformed_row)
        transformed_X = np.concatenate(transformed_X, axis=1)
        return transformed_X

    def inverse_transform(self, transformed_X):
        if transformed_X.ndim == 1:
            transformed_X = transformed_X.reshape(1, -1)
        X = []
        used_dim = 0
        for i in range(N_FEATURE):
            transformed_c = transformed_X[:, used_dim:used_dim + self.clip[i]].copy()
            for j in range(self.clip[i]):
                transformed_c[:, j] = transformed_c[:, j] / self.afterPODscale_list[i][j]
            original_row = self.pcas[i].inverse_transform(transformed_c)
            X.append(original_row)
            used_dim += self.clip[i]

        X = np.stack(X, axis=2)
        # for i in range(X.shape[0]):
        #     for j in range(X.shape[1]):
        #         for k in range(X.shape[2]):
        #             X[i][j][k] = X[i][j][k] * self.scale_list[k] + self.min_list[k]
        return X

def get_pca_model(raw_data = None, n_components = None) -> MyPca:
    pca_fname = "{0}_{1}_pca.pickle".format(DEFAULT_FILE_PREFIX, n_components)
    try:
        with open(pca_fname, 'rb') as f:
            pca = pickle.load(f)
            print("pca model loaded from ", pca_fname)
    except FileNotFoundError:
        if raw_data is None:
            print("pca file hasn't generated.")
            exit(-1)
        pca = MyPca(n_components)
        pca.fit(raw_data[500:])
        with open(pca_fname, 'wb') as f:
            pickle.dump(pca, f)
    return pca

def calculate_transformed_data(pca, data, figure = False, plot_framework = None):
    tmp_data = []
    for i in range(len(data)):
        transformed_data_row = pca.transform(data[i])
        tmp_data.append(transformed_data_row)
        if figure and i % 200 == 0 and plot_framework is not None:
            # 抽查PCA
            plot_original_z = np.reshape(data[i][-1], (-1, N_FEATURE))[:, 0]
            levels = np.linspace(min(plot_original_z), max(plot_original_z), 14)
            plot_from_point_list(plot_framework, plot_original_z, levels=levels, name="step{0}".format(i))
            plot_transformed_z = np.reshape(pca.inverse_transform(transformed_data_row)[-1], (-1, N_FEATURE))[:, 0]
            plot_from_point_list(plot_framework, plot_transformed_z, levels=levels, name="inverse_pod{0}".format(i))
            pass
    transformed_data = np.array(tmp_data)
    return transformed_data

if __name__ == '__main__':
    start_time = time.time()
    frame_shape = get_frame_shape()
    fw = Framework(frame_shape[:, 0], frame_shape[:, 1])
    data, raw_data = get_data(n_seq=N_SEQ, max_file_read=1500, generate_from=400)
    n_components = min(6, min(len(raw_data), len(raw_data[0])))
    pca = get_pca_model(raw_data = raw_data, n_components = n_components)
    print("Get data time usage: ", time.time() - start_time)
    model = get_compiled_model(input_shape=(N_SEQ - 1, n_components))
    # model.save("nothing.h5")
    transformed_cache_fname = "transformed_{0}_{1}_{2}.pickle".format(N_SEQ, DEFAULT_FILE_PREFIX, n_components)
    try:
        with open(transformed_cache_fname, 'rb') as f:
            transformed_data = pickle.load(f)
            print("transformed data loaded from cache.")
    except FileNotFoundError:
        transformed_data = calculate_transformed_data(pca, data, plot_framework=fw, figure=True)
        print("transformed data calculated.")
        with open(transformed_cache_fname, 'wb') as f:
            pickle.dump(transformed_data, f)
    split_index = int(len(transformed_data) * 0.9)
    x_train = transformed_data[:split_index, :-1]
    y_train = transformed_data[:split_index, -1]
    x_test = transformed_data[split_index:, :-1]
    y_test = transformed_data[split_index:, -1]
    try:
        os.mkdir("checkpoints")
    except FileExistsError:
        pass
    CHECKPOINT_PATH = "checkpoints/{0}-cylinder-latest.hdf5".format(DEFAULT_FILE_PREFIX)

    cbs = [
        keras.callbacks.EarlyStopping(patience=430),
        keras.callbacks.TensorBoard(),
        keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, period=20, verbose=10,save_best_only=True),
        keras.callbacks.LearningRateScheduler(lambda x, y : (0.1 if x < 100 else (0.01 if x < 200 else (0.001 if x < 300 else 0.1 ** (x / 75.)))))
    ]

    try:
        model.load_weights(CHECKPOINT_PATH)
    except OSError:
        pass
    except ValueError:
        os.remove(CHECKPOINT_PATH)

    # while True:
    #     if os.path.exists(DEFAULT_FILE_PREFIX + "last.wt"):
    #         model.load_weights(DEFAULT_FILE_PREFIX + "last.wt")
    transformed_raw_fname = "transformed_raw_{0}_{1}_{2}.pickle".format(N_SEQ, DEFAULT_FILE_PREFIX, n_components)
    try:
        with open(transformed_raw_fname, 'rb') as f:
            transformed_raw_data = pickle.load(f)
    except FileNotFoundError:
        transformed_raw_data = pca.transform(raw_data)
        with open(transformed_raw_fname, 'wb') as f:
            pickle.dump(transformed_raw_data, f)
    print(transformed_raw_data.shape)
    for i in range(n_components):
        plt.plot(transformed_raw_data[:, i], label="POD value No.{0}".format(i))
    plt.title("the first {0} POD values by timestep".format(n_components))
    plt.legend()
    plt.show()
    train_start_time = time.time()
    model.fit(x_train, y_train, batch_size=64, epochs=10000, callbacks=cbs, validation_data=(x_test, y_test))
    print("Train finished, time usage: ", time.time() - train_start_time)
    # for i in range(10):
    #     internal_data = np.array(predict_internal_data(model, transformed_raw_data[:split_index], n_seq=N_SEQ))
    #     print(internal_data.shape)
    #     # transformed_internal_data = calculate_transformed_data(pca, internal_data)
    #     x_train = internal_data[:,:-1]
    #     y_train = internal_data[:,-1]
    #     model.fit(x_train, y_train, batch_size=64, epochs=10000, callbacks=cbs, validation_data=(x_test, y_test))
    errors = [0 for _ in range(N_SEQ)]
    for i in range(10):
        test_index = random.randint(0, x_train.shape[0] - 1)
        test_data = x_train[test_index: test_index + 1]
        test_target = y_train[test_index: test_index + 1]
        result = model.predict(test_data)
        plot_original_z = np.reshape(data[test_index][-1], (-1, N_FEATURE))[:, 0]
        levels = np.linspace(min(plot_original_z), max(plot_original_z), 14)
        plot_from_point_list(fw, plot_original_z, levels=levels, name="true_pic"+time.strftime("%Y-%m-%d"))
        plot_transformed_z = np.reshape(pca.inverse_transform(result), (-1, N_FEATURE))[:, 0]
        plot_from_point_list(fw, plot_transformed_z, levels=levels, name="predicted_pic"+time.strftime("%Y-%m-%d"))
        mse = np.mean(np.square(plot_original_z - plot_transformed_z))
        print("step[{}]'s mse: {}".format(i, mse))
        errors.append(mse)
