import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# given a text file containing distances, create x and y values for plotting on a graph
THRESHOLD = 10
with open('distance.txt') as f:
    data = f.readlines()
    data = [float(x.strip()) for x in data]
    x = np.arange(0, len(data), 1)

    arr = np.array(data).reshape(-1, 1)
    arr = arr[arr[:, 0] < 1500]
    max_point = max(arr) - THRESHOLD
    min_point = min(arr) - THRESHOLD
    print(min_point, max_point)
    x_1 = np.arange(0, len(arr), 1)
    plt.scatter(x_1, arr, label='distance', c='g')

plt.legend()
plt.show()
