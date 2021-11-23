import copy
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
EPSILON = 1e-7

#MSLELoss clone https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/keras/losses/MSLE
class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        pred = torch.clamp(pred, min=EPSILON, max=None)
        actual = torch.clamp(actual, min=EPSILON, max=None)
        return self.mse(torch.log2(pred + 1), torch.log2(actual + 1))

class MAPELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        pred = torch.clamp(pred, min=1, max=None)
        actual = torch.clamp(actual, min=1, max=None)
        diff = torch.abs(torch.log2(actual+1)-torch.log2(pred+1)) / torch.log2(actual+2)
        loss = torch.mean(diff)
        return loss


class MyDataset(Dataset):
    def __init__(self, input_vae, input_global, label, max_length):
        self.vae, self.input_global, self.y = input_vae, input_global, label
        self.max_cascade_length = max_length
        self.len = len(input_vae)

    def __len__(self):
        return self.len

    def __getitem__(self, x_idx):
        b_vae = self.vae[x_idx]
        b_global = self.input_global[x_idx]
        b_y = self.y[x_idx]

        b_time = np.array(b_vae)
        b_time = b_time[:, 0]
        b_time = b_time
        #判断有没有重复的时间
        time2 = np.insert(b_time, 0, 0)
        time3 = time2[0 : -1]
        temp = b_time - time3
        idx = np.where(temp == 0)


        if len(idx[0]):
            idx = np.array(idx)
            idx = np.squeeze(idx, 0)
            idx_next = idx + 1
            idx = idx.tolist()
            idx_next = idx_next.tolist()

            if idx_next[-1] == len(b_time):
                b_time[idx[-1]] = b_time[idx[-1]] - 0.001
                b_time[idx[0:-1]] = (b_time[idx[0:-1]] + b_time[idx_next[0:-1]]) / 2.0
            else:
                b_time[idx[0:]] = (b_time[idx[0:]] + b_time[idx_next[0:]]) / 2.0

            # for i in range(0, len(idx)):
            #     a = idx[i]
            #     if a == len(b_time)-1:
            #         b_time[a] = b_time[a]- 0.001
            #     else:
            #         b_time[a] = (b_time[a] + b_time[a+1]) / 2.0

            # b_time[0] = (b_time[0] + b_time[1]) / 2.0


        b_time = b_time.tolist()


        while len(b_vae) < self.max_cascade_length:
            b_vae.append(np.zeros(shape=len(b_vae[0])))
        while len(b_global) < self.max_cascade_length:
            b_global.append(np.zeros(shape=len(b_global[0])))

        while len(b_time) < self.max_cascade_length:
            b_time.append(b_time[-1]-0.001)

        b_x = np.concatenate([np.array(b_vae), np.array(b_global)], axis=1)

        return b_x, np.array(b_y), np.array(b_time)


class Sampling(nn.Module):
    def __init__(self):
        super(Sampling, self).__init__()

    def forward(self, mean, log_var):
        mean = mean
        log_var = log_var
        batch = mean.shape[0]
        dim = mean.shape[1]

        EPSILON = np.random.randn(batch, dim)
        EPSILON = torch.from_numpy(EPSILON)
        EPSILON = EPSILON.to(torch.float32)
        EPSILON = EPSILON.to('cuda:0')

        # EPSILON = torch.randn(batch, dim)
        # EPSILON = EPSILON.to('cuda:0')
        return mean + torch.exp(.5 * log_var) * EPSILON

