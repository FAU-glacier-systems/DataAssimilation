import matplotlib.pyplot as plt
import numpy as np
from numpy.random import multivariate_normal


cov = np.zeros((6,6))
sigma = multivariate_normal([1, 2, 3, 4, 5, 6 ], cov, 5)
print(sigma)
