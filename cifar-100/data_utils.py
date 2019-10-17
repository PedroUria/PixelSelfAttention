import pickle
import numpy as np
import torch

def load_data(path):
    with open(path, 'rb') as s:
        data_dict = pickle.load(s, encoding='bytes')
    x = []
    for img in data_dict[b"data"]:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[..., 0] = np.reshape(img[:1024], (32, 32))  # Red channel
        image[..., 1] = np.reshape(img[1024:2048], (32, 32))  # Green channel
        image[..., 2] = np.reshape(img[2048:], (32, 32))  # Blue channel
        x.append(image)
    return np.array(x), np.array(data_dict[b"fine_labels"])

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
