from rbf.interpolate import RBFInterpolant
from scipy.interpolate import Rbf
from network import *
import numpy as np
import pyprind

if __name__ == '__main__':
    start_time = time.time()
    frame_shape = get_frame_shape()
    fw = Framework(frame_shape[:, 0], frame_shape[:, 1])
    data, raw_data = get_data(n_seq=2, max_file_read=1000, generate_from=500)
    n_components = min(3, min(len(raw_data), len(raw_data[0])))
    pca = get_pca_model(raw_data = raw_data, n_components = n_components)
    print("Get data time usage: ", time.time() - start_time)

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
    transformed_raw_fname = "transformed_raw_{0}_{1}_{2}.pickle".format(N_SEQ, DEFAULT_FILE_PREFIX, n_components)
    try:
        with open(transformed_raw_fname, 'rb') as f:
            transformed_raw_data = pickle.load(f)
    except FileNotFoundError:
        transformed_raw_data = pca.transform(raw_data)
        with open(transformed_raw_fname, 'wb') as f:
            pickle.dump(transformed_raw_data, f)
    print(transformed_raw_data.shape)
    plt.plot(transformed_raw_data[:, 0])
    plt.title("第一个特征值随时间变化曲线图")
    plt.show()

    rbf_model = [Rbf(data[:, 0, 0], data[:, 0, 1], data[:, 0, 2], data[:, 1, i]) for i in pyprind.prog_bar(range(n_components), title="generating RBFInterpolant")]
    print("RBF model generated.")
    for i in range(500, 999):
        predicted = []
        for j in range(n_components):
            predicted.append(rbf_model[j](transformed_raw_data[i][0], transformed_raw_data[i][1], transformed_raw_data[i][2]))
        np.stack(predicted, axis=0)
        print(predicted)




