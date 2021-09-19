import numpy as np
import matplotlib.pyplot as plt rand = np.random
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 7)) i0input1=rand.normal(8,1.5,20) i0input2=rand.normal(8,1.5,20) i1input1=rand.normal(15,1.5,20) i1input2=rand.normal(15,1.5,20)

plt.subplot(2, 2,1)
plt.title('instance 0 input 1') plt.hist(i0input1, bins=50,color='red')

plt.subplot(2, 2,2)
plt.title('instance 0 input 2') plt.hist(i0input2, bins=50,color='yellow')

plt.subplot(2, 2,3)
plt.title('instance 1 input 1') plt.hist(i1input1, bins=50,color='purple')

plt.subplot(2, 2,4)
plt.title('instance 1 input 2') plt.hist(i1input2, bins=50,color='green')

plt.show()
