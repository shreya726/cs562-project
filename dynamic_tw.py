from dtw import dtw
import numpy as np

def dynamic_time_warping(data1, data2):
	dist, cost, acc, path = dtw(x, y, dist=lambda x, y: np.linalg.norm(x - y, ord=1))
	print('Min distance found: ', dist)

# For testing
# Based on the example found here:  https://github.com/pierre-rouanet/dtw/blob/master/examples/simple%20example.ipynb
x = np.array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
y = np.array([1, 1, 1, 2, 2, 2, 2, 3, 2, 0]).reshape(-1, 1)