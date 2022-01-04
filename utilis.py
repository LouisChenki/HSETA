import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random


def list2padded(trip_list, embeddings):
    index = sorted(range(len(trip_list)), key=lambda k: len(trip_list[k]))
    index = index[::-1]
    trip_list_sorted = list()
    lengths = list()
    for i in index:
        lengths.append(len(trip_list[i]))
    index, lengths = np.array(index), np.array(lengths)
    index_index = np.where((lengths > 5) & (lengths < 700))
    index = index[index_index]
    lengths = lengths[index_index]
    for i in index:
        trip_list_sorted.append(torch.tensor(trip_list[i]))
    np.save(r'/mnt/DataDisk/GisCup2021/giscup_2021/results/dataset/index.npy', index)
    np.save(r'/mnt/DataDisk/GisCup2021/giscup_2021/results/dataset/lengths.npy', lengths)
    trip_list_sorted = torch.load(r'/mnt/DataDisk/GisCup2021/giscup_2021/results/dataset/trip_list_sorted.pt')
    trip_list_sorted = trip_list_sorted[:10000]
    trip_padded = pad_sequence(trip_list_sorted)
    trip_embedding = embeddings(trip_padded)
    return trip_embedding


def list2padded_2(trip_list, embeddings):
    """
    For the second data_process
    :param trip_list:
    :param embeddings:
    :return:
    """
    # index = sorted(range(len(trip_list)), key=lambda k: len(trip_list[k]))
    # index = index[::-1]
    # trip_list_sorted = list()
    # lengths = list()
    # for i in index:
    #     lengths.append(len(trip_list[i]))
    # index, lengths = np.array(index), np.array(lengths)
    # for i in index:
    #     trip_list_sorted.append(torch.tensor(trip_list[i]))
    # np.save(r'/mnt/FastData/GisCup2021/0610_/bridge_Data/trip_related/index.npy', index)
    # np.save(r'/mnt/FastData/GisCup2021/0610_/bridge_Data/trip_related/lengths.npy', lengths)
    # torch.save(trip_list_sorted, r'/mnt/FastData/GisCup2021/0610_/bridge_Data/trip_related/trip_list_sorted.pt')
    # trip_embedding = None
    trip_list_sorted = torch.load(r'/mnt/DataDisk/GisCup2021/giscup_2021/results/dataset/trip_list_sorted.pt')
    trip_list_sorted = trip_list_sorted[:10000]
    trip_padded = pad_sequence(trip_list_sorted)
    trip_embedding = embeddings(trip_padded)
    return trip_embedding


def part_embeddings(part, embeddings):
    trip_padded = pad_sequence(part)
    trip_embedding = embeddings(trip_padded)
    return trip_embedding



def matrix_sort(data, index):
    if len(list(data.shape)) > 1:
        data_sorted = data[index, :]
    else:
        data_sorted = data[index]
    return data_sorted


def pack_seq(batch_tensor, length):
    batch_tensor = pack_padded_sequence(batch_tensor, length, batch_first=False)
    return batch_tensor


def time_embeddings(time, d_model, period):
    pe = torch.zeros((period, d_model))
    pos = torch.arange(period).unsqueeze(1)
    pe[:, 0::2] = torch.sin(pos / torch.pow(1000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model))
    pe[:, 1::2] = torch.cos(pos / torch.pow(1000, torch.arange(1, d_model, 2, dtype=torch.float32) / d_model))
    pe = pe.T
    return pe[:, time].T


def read_tensor_list(root_path):
    tensor_list = list()
    for i in range(len(os.listdir(root_path))):
        tensor_list.append(torch.load(os.path.join(root_path, '{}.pt'.format(i))))
    return tensor_list


def label_mask(current_status, arrive_status):
    """
    To set the element of arrive status to 0 where the same place that the current status is 0
    :param current_status:
        Input tensor with shape [seq, batch_size, one-hot]
    :param arrive_status:
        Input tensor with shape [seq batch_size, one-hot]
    :return:
        The masked label tensor
    """
    current_status_v = torch.max(current_status, dim=-1)[0]
    arrive_status_v = torch.max(arrive_status, dim=-1)[0]
    # the index of label 1 and 0 all is 0, so to +1, but need to filter the 0 of arrive_status
    arrive_status_ = torch.max(arrive_status, dim=-1)[1] + 1
    current_status_v_arr = current_status_v.numpy()
    arrive_status_v_arr = arrive_status_v.numpy()
    arrive_status_arr = arrive_status_.numpy()
    arrive_status_arr[np.where(current_status_v_arr == 0)] = 0
    arrive_status_arr[np.where(arrive_status_v_arr == 0)] = 0
    arrive_status_tensor = torch.from_numpy(arrive_status_arr)
    return arrive_status_tensor


def org_rnn_input(x, deep_feature, wide_feature, length):
    """
    To repeat the deep_feature and wide_feature along the seq dimension
    and to be cated with x along feature dimension
    :param x: 
        Input tensor with shape [seq, batch_size, feature]
    :param deep_feature:
        Input tensor with shape [batch_size, feature_deep]
    :param
        Input tensor with shape [batch_size, feature_wide]
    :return:
        Output packed
    """
    seq, batch_size, feature = x.shape
    deep_feature = deep_feature.unsqueeze(0)
    wide_feature = wide_feature.unsqueeze(0)
    # *
    deep_feature = torch.repeat_interleave(deep_feature, seq, dim=0)
    wide_feature = torch.repeat_interleave(wide_feature, seq, dim=0)
    x = torch.cat([x, deep_feature, wide_feature], dim=-1)
    x = pack_seq(x, length)
    return x


def cat_global_feature(features: list):
    """
        To struct the global feature
    :param features:
        [dist: npy, simple: npy, time: tensor, week: tensor, weather: tensor]
    :return:
        Output tensor with shape [batch_size, features]
    """
    dist, simple, time, week, weather = features
    global_feature_list = list()
    for i in range(len(dist)):
        dist_in = torch.from_numpy(dist[i]).unsqueeze(1)
        simple_in = torch.from_numpy(simple[i]).unsqueeze(1)
        time_in = time[i]
        week_in = week[i]
        weather_in = weather[i]
        per_global_features = torch.cat([dist_in, simple_in, time_in, week_in, weather_in], dim=1)
        global_feature_list.append(per_global_features.float())
    return global_feature_list


def cat_global_feature_3(features: list):
    """
        To struct the global feature
    :param features:
        [dist: npy, simple: npy, time: tensor, week: tensor, weather: tensor, driver: tensor]
    :return:
        Output tensor with shape [batch_size, features]
    """
    dist, simple, time, week, weather, driver = features
    global_feature_list = list()
    for i in range(len(dist)):
        dist_in = torch.from_numpy(dist[i]).unsqueeze(1)
        simple_in = torch.from_numpy(simple[i]).unsqueeze(1)
        time_in = time[i]
        week_in = week[i]
        weather_in = weather[i]
        driver_in = driver[i]
        per_global_features = torch.cat([dist_in, simple_in, time_in, week_in, weather_in, driver_in], dim=1)
        global_feature_list.append(per_global_features.float())
    return global_feature_list


def fold_cross(index_len, fold, step):
    """
    To random slipt data according to cross
    :param index_len:
    :param fold:
    :return:
    """
    r = random.random
    random.seed(3)
    index = list(range(index_len))
    random.shuffle(index, random=r)
    index_index = list(range(0, index_len, int(index_len/fold)))
    index_index[-1] = index_len
    index_list = [index[i:i+int(index_len/fold)] for i in index_index[:-1]]
    test_index = index_list[step-1]
    training_index = [i for i in index if i not in test_index]
    return training_index, test_index


def rmse(out, labels):
    out = (out * (11747 - 8)) + 8
    labels = (labels * (11747 - 8)) + 8
    y = torch.sqrt(torch.mean(torch.pow(torch.abs(out - labels), 2)))
    return y


def mae(out, labels):
    out = (out * (11747 - 8)) + 8
    labels = (labels * (11747 - 8)) + 8
    y = torch.mean(torch.abs(out - labels))
    return y


def rmse_2(out, labels):
    out = (out * (3566 - 0)) + 0
    labels = (labels * (3566 - 0)) + 0
    y = torch.sqrt(torch.mean(torch.pow(torch.abs(out - labels), 2)))
    return y


def mae_2(out, labels):
    out = (out * (3566 - 0)) + 0
    labels = (labels * (3566 - 0)) + 0
    y = torch.mean(torch.abs(out - labels))
    return y


class ETAdataset(Dataset):
    def __init__(self, path_root, split, step):
        self.path_root = path_root
        self.trip_file_path = os.path.join(path_root, "trip_tensor_batch")
        self.ltime_file_path = os.path.join(path_root, "link_time_batch_norm")
        self.length = len(os.listdir(self.trip_file_path))
        self.index = None
        self.step = step
        self.index_list = self.fold_divide(split)

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        trip = torch.load(os.path.join(self.trip_file_path, str(self.index_list[index])+'.pt'))
        ltime = torch.load(os.path.join(self.ltime_file_path, str(self.index_list[index])+'.pt')).unsqueeze(-1)
        self.index = self.index_list[index]
        return trip, ltime, self.index

    def which_index(self):
        return self.index

    def fold_divide(self, split):
        r = random.random
        random.seed(3)
        index = list(range(self.length))
        random.shuffle(index, random=r)
        index_index = list(range(0, self.length, int(self.length / 6)))
        index_index[-1] = self.length
        index_list = [index[i:i + int(self.length / 6)] for i in index_index[:-1]]
        test_index = index_list[self.step - 1]
        training_index = [i for i in index if i not in test_index]
        if split == 'training':
            return training_index
        elif split == 'validation':
            return test_index



