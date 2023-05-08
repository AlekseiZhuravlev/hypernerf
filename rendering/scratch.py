import pickle

import numpy as np

points_path = '/home/azhuavlev/Desktop/Projects/Data/HyperNerf/hand1-dense-v2/points.npy'

points = np.load(points_path)
print(points.shape)

# plot points as 3d scatter plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_scatter(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    plt.show()

plot_3d_scatter(points)
