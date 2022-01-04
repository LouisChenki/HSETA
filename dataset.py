# Author:wuhuaisj
# Date:2021/6/22 8:52
# Github:https://github.com/chuwuhua

import os
import torch
import numpy as np
from torch.utils.data import Dataset


def cat_global_feature(features: list):
    dist, simple, time, week, weather, driver = features

    dist = torch.tensor(dist).unsqueeze(-1)
    simple = torch.tensor(simple).unsqueeze(-1)
    per_global_features = torch.cat([dist, simple, time, week, weather, driver], dim=1)

    return per_global_features


class DiDiData(Dataset):
    def __init__(self, source_data, index, device):
        super(DiDiData).__init__()
        self.data_root = source_data
        self.index = index
        self.device = device
        self.struct_path()
        self.load()
        print('***** init down *****')

    def load(self):
        """
        load 部分数据，还有部分数据在getitem中加载
        :return:
        """
        # load global feature
        print('***** load global feature *****')
        self.ata = np.load(os.path.join(self.global_feature,
                                        'ata_batch.npy'), allow_pickle=True)
        self.eta = np.load(os.path.join(self.global_feature,
                                        'eta_batch.npy'), allow_pickle=True)
        self.distance = np.load(os.path.join(
            self.global_feature, 'distance_batch.npy'), allow_pickle=True)
        self.lengthes = np.load(os.path.join(
            self.global_feature, 'length_batch.npy'), allow_pickle=True)
        self.slice_id = torch.load(os.path.join(
            self.global_feature, 'slice_id_batch.pt'))
        self.week = torch.load(os.path.join(
            self.global_feature, 'week_batch.pt'))
        self.weather = torch.load(os.path.join(
            self.global_feature, 'weather_batch.pt'))
        self.driver = torch.load(os.path.join(
            self.global_feature, 'driver_batch.pt'))
        self.link_time = torch.load(os.path.join(
            self.global_feature, 'link_time_batch.pt'))
        self.sparse_feature = torch.load(os.path.join(
            self.global_feature, 'sparse_batch.pt'))
        self.batch_num = len(self.ata)

    def struct_path(self):
        """
        构造路径
        :return:
        """
        print('***** struct path *****')
        self.trip_path = os.path.join(self.data_root, 'trip_batch')
        # self.link_time_path = os.path.join(self.data_root, 'link_time_batch')
        self.global_feature = os.path.join(self.data_root, 'global_feature')

    def __len__(self):
        return len(self.index)

    def __getitem__(self, item):
        item = self.index[item]
        trip = torch.load(os.path.join(self.trip_path, '{}.pt'.format(item)))
        # link_time = torch.load(os.path.join(self.link_time_path, '{}.pt'.format(item)))
        link_time = self.link_time[item]
        global_feature = cat_global_feature(
            [self.distance[item], self.eta[item], self.slice_id[item], self.week[item], self.weather[item],
             self.driver[item]])
        trip_feature = torch.cat(
            [trip, link_time], dim=-1).float()
        sparse_feature = self.sparse_feature[item]
        label = torch.tensor(self.ata[item]).unsqueeze(
            1).float()
        return global_feature, trip_feature, sparse_feature, self.lengthes[item], label


if __name__ == '__main__':
    data = DiDiData(source_data='../data/')
    print(len(data))
