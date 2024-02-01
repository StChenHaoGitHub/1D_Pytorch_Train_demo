import numpy as np


def package_dataset(data, label):
    dataset = [[i, j] for i, j in zip(data, label)]
    # channel number
    channels = data[0].shape[0]
    # data length
    length = data[0].shape[1]
    # data classes
    classes = len(np.unique(label))
    return dataset, channels, length, classes


if __name__ == '__main__':
    data = np.load('Dataset/data.npy')
    label = np.load('Dataset/label.npy')
    dataset, channels, length, classes = package_dataset(data, label)
    print(channels, length, classes)