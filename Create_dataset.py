import numpy as np

numbers = 100
channels = 3
length = 224
classes = 2

data = np.random.randn(numbers,channels,length)
label = np.random.randint(0,classes,numbers)

np.save('Dataset/data.npy',data,allow_pickle=True)
np.save('Dataset/label.npy',label,allow_pickle=True)
