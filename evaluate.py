# ./evaluate.py
from network import get_compiled_model, get_data, get_pca_model, get_raw_data, MyPca
from residual_network import get_compiled_residual_model
from util.constants import *
import pickle
from util.ploter import plot_from_point_list, Framework, get_frame_shape
import numpy as np
import matplotlib.pyplot as plt
import pyprind
CHECKPOINT_PATH = "checkpoints/{0}-cylinder-latest.hdf5".format(DEFAULT_FILE_PREFIX)

N_STEP = N_SEQ
N_PROP = N_FEATURE
n_component = 6
if __name__ == '__main__':

    model = get_compiled_model((N_STEP - 1, n_component))
    try:
        model.load_weights(CHECKPOINT_PATH)
    except OSError:
        pass

    pca = get_pca_model(n_components=n_component)
    raw_data = get_raw_data(max_file_read=2000)
    print("data loaded.")
    frame_shape = get_frame_shape()
    slide_window = []
    transformed_seq = pca.transform(raw_data)
    START_STEP = 0 // M_GEN
    STEADY_STEP = 1000 // M_GEN
    END_STEP = 2000 // M_GEN
    for i in range(START_STEP + N_STEP - 1):
        slide_window.append(transformed_seq[i])
    slide_window = np.array(slide_window)
    error_list = [0 for i in range(START_STEP + N_STEP - 1)] # first 100 step is accurate
    error_list = []
    fw = Framework(frame_shape[:, 0], frame_shape[:, 1])
    inspect_point_index = fw.findByXy(1.2, .5)
    inspect_value = [raw_data[i][inspect_point_index]for i in range(START_STEP + N_STEP - 1)]
    inspect_value_exact = [raw_data[i][inspect_point_index] for i in range(END_STEP)]
    for i in pyprind.prog_bar(range(START_STEP + N_STEP, END_STEP), stream=1):
        if i > STEADY_STEP and i % 10 != 0:
            new_row = model.predict(np.reshape(slide_window[i - N_STEP: i - 1], (1, N_STEP - 1, n_component)))
        else:
            new_row = model.predict(np.reshape(transformed_seq[i - N_STEP: i - 1], (1, N_STEP - 1, n_component)))

        slide_window = np.append(slide_window, new_row, axis=0)
        plot_original_z = np.reshape(raw_data[i], (-1, N_FEATURE))[:, 0]
        plot_transformed_z = np.reshape(pca.inverse_transform(np.reshape(new_row, (-1,))), (-1, N_FEATURE))[:, 0]
        inspect_value.append(plot_transformed_z[inspect_point_index])
        if i % 400 == 0:
            levels = np.linspace(min(plot_original_z) - 0.3, max(plot_original_z) + 0.3, 14)
            plot_from_point_list(fw, plot_original_z, levels=levels, name="true_pic_step["+ str(i) + "]")
            plot_from_point_list(fw, plot_transformed_z, levels=levels, name="predicted_pic_step[" + str(i) + "]")
        mse = np.mean(np.square(plot_original_z - plot_transformed_z))
        error_list.append(mse)
        # print("step {}'s mse {}".format(i, mse))
    with open("error_list_u1_5000.pickle", 'wb') as f:
        pickle.dump(error_list, f)
    plt.plot(inspect_value_exact, '--')
    plt.plot(inspect_value)
    plt.show()
    plt.clf()
    plt.plot(error_list)
    plt.show()
