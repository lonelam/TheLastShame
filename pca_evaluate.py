from network import get_pca_model

import numpy as np

class MyPOD():
    def __init__(self):
        pass
    def fit(self, data: list):
        self.data = np.array(data)
        np.linalg.svd()

if __name__ == '__main__':
