from scipy.interpolate import Rbf
import matplotlib.p
import  numpy as np

x = y = np.linspace(0, 1, 10)
z = x * y

rbfi = Rbf(x, y, z)
xi = yi = np.linspace(0, 1, 20)
xygrid = np.meshgrid(xi, yi)
print(rbfi(xi, yi))