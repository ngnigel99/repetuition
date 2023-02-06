import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from src.custom_nodes.dabble.repetuition import depth_file_denoizer
# given a text file containing distances, create x and y values for plotting on a graph

with open('distance.txt') as f:
    data = f.readlines()
    data = [float(x.strip()) for x in data]
    x = np.arange(0, len(data), 1)
    plt.scatter(x, data, label='distance', c='g')

maxdepth, mindepth = depth_file_denoizer('distance.txt')

data_within_range = []
x_within_range = []

for i in range(len(data)):
    if mindepth <= data[i] <= maxdepth:
        data_within_range.append(data[i])
        x_within_range.append(x[i])

plt.scatter(x_within_range, data_within_range, label='de-noised distance', c='b')

plt.axhline(y = maxdepth)
plt.axhline(y = mindepth)

plt.ylabel("pixel distance")
plt.xlabel("sample number")

plt.legend()
plt.show()
