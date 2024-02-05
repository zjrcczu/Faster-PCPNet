import argparse

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

import pickle as pkl

import cv2


class dataset(Dataset):
    def __init__(self, cfg, mode='train'):
        super(dataset, self).__init__()
        self.cfg = cfg
        self.mode = mode
        data_path = cfg.data_path + '\\{}_data.pkl'.format(mode)
        with open(data_path, 'rb') as f:
            data = pkl.load(f)
        self.data = data
        if cfg.balance and mode == 'train':
            self.balance_data()

        crossing = self.data['crossing']
        print('crossing: ', crossing.shape)
        print('crossing 0: ', np.sum(crossing == 0))
        print('crossing 1: ', np.sum(crossing == 1))

    def __getitem__(self, index):
        image = self.data['image'][index]
        box = np.copy(self.data['box'][index])
        trajectory = np.copy(self.data['center'][index])
        joint = np.copy(self.data['joint'][index])
        speed = np.copy(self.data['speed'][index])
        crossing = self.data['crossing'][index]

        # box[:, 0] = box[:, 0] / 1920
        # box[:, 1] = box[:, 1] / 1080
        # box[:, 2] = box[:, 2] / 1920
        # box[:, 3] = box[:, 3] / 1080

        # O2
        angle = trajectory - np.array([960, 1080])
        angle = angle[:, 0] / np.sqrt(np.sum(np.square(angle), axis=1))
        angle = torch.from_numpy(angle).float().unsqueeze(-1)
        # O1
        angle1 = trajectory - np.array([0, 1080])
        angle1 = angle1[:, 0] / np.sqrt(np.sum(np.square(angle1), axis=1))
        angle1 = torch.from_numpy(angle1).float().unsqueeze(-1)
        # O3
        angle2 = trajectory - np.array([1920, 1080])
        angle2 = angle2[:, 0] / np.sqrt(np.sum(np.square(angle2), axis=1))

        # only use angle
        angle2 = torch.from_numpy(angle2).float().unsqueeze(-1)

        angle = torch.cat((angle, angle1, angle2), dim=1)
        # O2
        dist = np.abs(trajectory - np.array([960, 1080]))
        dist = np.sqrt(np.sum(np.square(dist), axis=1)) / 2200
        dist = torch.from_numpy(dist).float().unsqueeze(-1)
        # O1
        dist1 = np.abs(trajectory - np.array([0, 1080]))
        dist1 = np.sqrt(np.sum(np.square(dist1), axis=1)) / 2200
        dist1 = torch.from_numpy(dist1).float().unsqueeze(-1)
        # O3
        dist2 = np.abs(trajectory - np.array([1920, 1080]))
        dist2 = np.sqrt(np.sum(np.square(dist2), axis=1)) / 2200
        dist2 = torch.from_numpy(dist2).float().unsqueeze(-1)

        dist = torch.cat((dist, dist1, dist2), dim=1)

        # use angle and dist
        dist = torch.cat((dist, angle), dim=1)
        dist = torch.cat((torch.zeros(1, 6), dist[1:, :] - dist[:-1, :]), dim=0)

        # dist = dist[:, [0, 1, 4, 5]]

        seg = torch.zeros(1, 256, 192)

        width = box[:, 2] - box[:, 0]
        height = box[:, 3] - box[:, 1]
        ratio = (width / height) * 100
        area = (width * height / (1920 * 1080)) * 10000

        width = width[:, np.newaxis] / 1920
        height = height[:, np.newaxis] / 1080

        wh = np.concatenate((width, height), axis=1)

        trajectory[:, 0] = trajectory[:, 0] / 1920
        trajectory[:, 1] = trajectory[:, 1] / 1080

        joint[:, :, 0] = joint[:, :, 0] / 1920
        joint[:, :, 1] = joint[:, :, 1] / 1080

        area = torch.from_numpy(area[0:self.cfg.observe_time]).float().unsqueeze(-1)

        ratio = torch.from_numpy(ratio).float().unsqueeze(-1)

        trajectory = torch.from_numpy(trajectory).float()

        trajectory = torch.cat((trajectory, ratio), dim=1)
        joint = torch.from_numpy(joint[0:self.cfg.observe_time, :, 0:self.cfg.kps_ch]).float()
        speed = torch.from_numpy(speed[0:self.cfg.observe_time, :]).float()
        dist = dist[0:self.cfg.observe_time, :].float()
        wh = torch.from_numpy(wh[0:self.cfg.observe_time, :]).float()
        crossing = torch.from_numpy(crossing).long()

        # return [box, trajectory, joint, speed, segment, crossing, action]
        return [joint, speed, trajectory, area, seg, dist, angle, wh, crossing]

    def __len__(self):
        return self.data['center'].shape[0]

    def balance_data(self):
        # data = np.copy(self.data)
        pos_index = np.where(self.data['crossing'] == 1)[0]
        neg_index = np.where(self.data['crossing'] == 0)[0]
        pos_num = len(pos_index)
        neg_num = len(neg_index)
        diff = pos_num - neg_num
        if diff > 0:
            pos_index = np.random.choice(pos_index, neg_num * 2, replace=False)
            index = np.concatenate((pos_index, neg_index), axis=0)
            index.sort()
        elif diff < 0:
            neg_index = np.random.choice(neg_index, pos_num * 2, replace=False)
            index = np.concatenate((pos_index, neg_index), axis=0)
            index.sort()
        for key in self.data.keys():
            self.data[key] = self.data[key][index]
        pos_index = np.where(self.data['crossing'] == 1)[0]
        for i in pos_index:
            box = np.copy(self.data['box'][i])
            box[:, 0] = 1920 - box[:, 2]
            box[:, 2] = 1920 - box[:, 0]
            trajectory = np.copy(self.data['center'][i])
            trajectory[:, 0] = 1920 - trajectory[:, 0]
            joint = np.copy(self.data['joint'][i])
            joint[:, :, 0] = 1920 - joint[:, :, 0]
            crossing = np.copy(self.data['crossing'][i])
            speed = np.copy(self.data['speed'][i])
            self.data['box'] = np.concatenate((self.data['box'], box[np.newaxis, :]), axis=0)
            self.data['center'] = np.concatenate((self.data['center'], trajectory[np.newaxis, :]), axis=0)
            self.data['joint'] = np.concatenate((self.data['joint'], joint[np.newaxis, :]), axis=0)
            self.data['crossing'] = np.concatenate((self.data['crossing'], crossing[np.newaxis, :]), axis=0)
            self.data['speed'] = np.concatenate((self.data['speed'], speed[np.newaxis, :]), axis=0)
        print(len(index))
        print(np.where(self.data['crossing'] == 1)[0].shape[0])
        print(np.where(self.data['crossing'] == 0)[0].shape[0])
        # box = data['box']
        # box[:, :, 0] = 1920 - box[:, :, 2]
        # box[:, :, 2] = 1920 - box[:, :, 0]
        # trajectory = data['center']
        # trajectory[:, :, 0] = 1920 - trajectory[:, :, 0]
        # joint = data['joint']
        # joint[:, :, :, 0] = 1920 - joint[:, :, :, 0]


if __name__ == '__main__':
    cfg = argparse.Namespace(**yaml.load(open('../config/pie_config.yaml', 'r'), Loader=yaml.FullLoader))
    cfg.seg = True
    train_dataset = dataset(cfg, mode='train')
    a = train_dataset.__getitem__(1)
    print(len(a))