import numpy as np

data = np.array(np.ones((1, 1024)), dtype=np.float32)
print(data)

np.save('1_1024.npy', data)

#data2 = np.load('5_5.npy')
