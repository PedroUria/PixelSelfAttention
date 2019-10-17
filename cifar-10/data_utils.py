import pickle
import numpy as np
import torch

def unpickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
    return data


def load_cifar10_data(data_dir):
    # https://luckydanny.blogspot.com/2016/07/load-cifar-10-dataset-in-python3.html
    train_data = None
    train_labels = []

    for i in range(1, 6):
        data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            train_data = data_dic['data']

        else:
            train_data = np.vstack((train_data, data_dic['data']))
        train_labels += data_dic['labels']

    test_data_dic = unpickle(data_dir + "/test_batch")
    test_data = test_data_dic['data']
    test_labels = test_data_dic['labels']

    train_data = train_data.reshape((len(train_data), 3, 32, 32))
    train_data = np.rollaxis(train_data, 1, 4)
    train_labels = np.array(train_labels)

    test_data = test_data.reshape((len(test_data), 3, 32, 32))
    test_data = np.rollaxis(test_data, 1, 4)
    test_labels = np.array(test_labels)

    return train_data, train_labels, test_data, test_labels

def prep_data(x, y, only_batch_in_gpu=True, use_cls=False):
    x, y = torch.FloatTensor(x) / 255, torch.LongTensor(y)
    if use_cls:
        p = torch.arange(x.shape[1] * 2 + 1, dtype=torch.long).unsqueeze(0).expand(size=(len(x), x.shape[1] * 2 + 1))
        s = torch.cat([torch.zeros((len(x), 1)), torch.ones((len(x), x.shape[1])),
                       2*torch.ones((len(x), x.shape[1]))], dim=1).long()
    else:
        p = torch.arange(x.shape[1] * 2, dtype=torch.long).unsqueeze(0).expand(size=(len(x), x.shape[1] * 2))
        s = torch.cat([torch.zeros((len(x), x.shape[1])), torch.ones((len(x), x.shape[1]))], dim=1).long()

    x.requires_grad = True
    if not only_batch_in_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x, p, s, y = x.to(device), p.to(device), s.to(device), y.to(
            device)
    return x, p, s, y
