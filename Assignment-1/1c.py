import numpy as np
from sklearn import datasets import matplotlib.pyplot as plt

X, y = datasets.make_circles(200, noise=0, random_state=42) X1, y1 = datasets.make_moons(200, noise=0, random_state=42) X2, y2= datasets.make_blobs(200, random_state=42)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 7)) plt.xlabel("x0", fontsize=10)
plt.ylabel("y0", fontsize=10) plt.subplot(1,3,1)
plt.scatter(X[:,0], X[:,1], s=60, c=y)

plt.xlabel("x1", fontsize=10) plt.ylabel("y1", fontsize=10) plt.subplot(1,3,2)
plt.scatter(X1[:,0], X1[:,1], s=60, c=y1)

plt.xlabel("x2", fontsize=10) plt.ylabel("y2", fontsize=10) plt.subplot(1,3,3)
plt.scatter(X2[:,0], X2[:,1], s=60, c=y2)


plt.show()
