import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Given two text files, plot the data in them on the same graph

proper_mean = 0
proper_std = 0

# get the sample mean and standard deviations of proper data
with open('proper_angle.txt') as f:
    data = f.readlines()
    # only take data if x >= 110 and x <= 210
    data = [float(x.strip()) for x in data]
    data = np.array(data).reshape(-1, 1)

    sta = StandardScaler()
    data = sta.fit_transform(data)
    # restrict y values to be between -1.5 and 1.5 by observing the data
    data = data[data[:, 0] >= -1.5]
    data = data[data[:, 0] <= 1.5]

    max_point = np.max(data)
    min_point = np.min(data)
    print("max_point = ", max_point)
    print("min_point = ", min_point)

    x = np.arange(0, len(data), 1)

    # get range of clusters
    plt.scatter(x, data, label='proper_angle', c='g')


with open('hindu_pushup.txt') as f:
    data = f.readlines()
    data = [float(x.strip()) for x in data]
    data = np.array(data).reshape(-1, 1)
    sta = StandardScaler()
    data = sta.fit_transform(data)
    
    #restrict data to -1.5, 1.5
    np.clip(data, -1.5, 1.5, out=data)
    # restrict values to be more than max_point or smaller than min_point
    data_1 = data[data[:, 0] >= max_point]

    # and greater tan -1.5
    data2 = data[data[:, 0] <= min_point]
    x = np.arange(0, len(data_1), 1)

    plt.scatter(x, data_1, label='hindu_pushup', c='r')


    x_2 = np.arange(0, len(data2), 1)
    plt.scatter(x_2, data2, label='hindu_pushup', c='b')
# add axis for plot
plt.xlabel('x - axis')
plt.ylabel('y - axis')

plt.legend()
plt.show()