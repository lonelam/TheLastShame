import numpy as np
from util.constants import *
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import csv
import os

def parse_ansys_shape(file):
    data = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            try:
                data.append([float(line[1]), float(line[2])])
            except ValueError:
                continue
            except IndexError:
                continue
    return data

def get_frame_shape(file_name = None):
    if file_name is None:
        for f in (f for f in os.listdir(FILE_PATH) if f.startswith(DEFAULT_FILE_PREFIX)):
            print("Parsing", f)
            raw_data_row = parse_ansys_shape(os.path.join(FILE_PATH, f))
            return np.array(raw_data_row)

def parse_ansys_export(file):
    data = []
    with open(file, 'r') as f:
        reader = csv.reader(f,)
        for line in reader:
            try:
                data.append([float(line[3]), float(line[4]), ])
            except ValueError:
                continue
            except IndexError:
                continue
    return data

class Framework:
    def __init__(self, X, Y, X_N = 200, Y_N = 200):
        minx = min(X)
        maxx = max(X)
        miny = min(Y)
        maxy = max(Y)
        element_list = list(zip(X, Y))
        self.kdt = KDTree(element_list)
        print('element number: ', len(element_list))
        self.plot_x = np.linspace(minx, maxx, X_N)
        self.plot_y = np.linspace(miny, maxy, Y_N)
        self.plot_z = np.zeros((X_N, Y_N))
    def updateZ(self, Z):
        dist = []
        self.cache_Z = Z
        for j in range(len(self.plot_y)):
            for i in range(len(self.plot_x)):
                near_point, near_id = self.kdt.query([self.plot_x[i], self.plot_y[j]])
                if self.plot_x[i] ** 2 + self.plot_y[i] ** 2 <= 0.25:
                    self.plot_z[j][i] = -1.
                else:
                    self.plot_z[j][i] = Z[near_id]
                dist.append(near_point)
        print(np.mean(dist))

    def plot(self, levels, name):
        fig, ax = plt.subplots()
        if levels is None:
            cset = ax.contourf(self.plot_x, self.plot_y, self.plot_z, 14, cmap='rainbow')
        else:
            cset = ax.contourf(self.plot_x, self.plot_y, self.plot_z, levels, cmap='rainbow')
        # ax.scatter(raw_data_x, raw_data_y, c=raw_data_z, cmap='rainbow')
        fig.colorbar(cset)
        ax.axis('equal')
        ax.set_title(name)
        fig.savefig(name + ".jpg")
        fig.show()

    def findByXy(self, x, y):
        return self.kdt.query([x, y])[1]



def plot_from_point_list(framework, Z, levels = None, name="default"):
    framework.updateZ(Z)
    framework.plot(levels, name)




if __name__ == '__main__':
    raw_data = parse_ansys_export("./exports/u1-8876.txt")
    raw_data_z = [row[1] for row in raw_data]
    shape = get_frame_shape()
    fw = Framework(shape[:, 0], shape[:, 1])
    plot_from_point_list(fw, raw_data_z)
