import torch
import torch.optim as optim
from torch.utils.data.dataset import T
from HSETA_related.HSETA.dataset import DiDiData
from HSETA_related.HSETA.hseta import HSETA, mape, NoamOpt
from HSETA_related.HSETA.utilis import fold_cross, rmse, mae
from torch.utils.data import DataLoader
import time
import numpy as np

fold = 6  # 交叉验证分段数
use_fold = 1
lr = 0.0005
epoch = 150
data_length = 8443  # batch数量
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


for use_fold in range(1, 7):
    train_index, test_index = fold_cross(data_length, fold=fold, step=use_fold)  # step 1-6

    train_dataset = DiDiData(source_data=r'/mnt/FastData/GisCup2021/0715_/train_processed_1024/', index=train_index, device=device)
    test_dataset = DiDiData(source_data=r'/mnt/FastData/GisCup2021/0715_/train_processed_1024/', index=test_index, device=device)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=None, num_workers=16, prefetch_factor=32,
                                  persistent_workers=True)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=None, num_workers=16, prefetch_factor=32,
                                 persistent_workers=True)


    net = HSETA(in_w=22, in_d=32, in_r=33, v_w=8, hidden_w=16, hidden_d=128, out_w=32, out_d=128, out_r=128)
    net.reset_parameters()
    net.to(device)
    print('***** using device {} *****'.format(device))


    criterion_regression = mape()

    # optimizer
    optimizer = optim.AdamW(net.parameters(), lr=lr)
    optimizer_w = NoamOpt(d_model=64, factor=1, warmup=20 * data_length, optimizer=optimizer)

    best_loss = float('inf')  # 确定存储哪一组参数

    for e in range(epoch):
        epoch_start = time.time()
        running_loss = 0
        net.train()
        start_time = time.time()
        for step, data in enumerate(train_dataloader):
            global_feature, trip_feature, sparse_feature, lengthes, label = data
            global_feature = global_feature.to(device)
            trip_feature = trip_feature.to(device)
            sparse_feature = sparse_feature.to(device)
            label = label.to(device)
            optimizer_w.zero_grad()
            outs = net(sparse_feature, global_feature, trip_feature, lengthes)
            loss = criterion_regression(outs, label)
            loss.backward()
            optimizer_w.step()

            c_step, c_lr = optimizer_w.qurey()

            running_loss = running_loss + loss.item()
            l2 = rmse(outs, label)
            l3 = mae(outs, label)

            if step % 1000 == 999:
                end_time = time.time()
                print(
                    "fold: {}, epoch: {}, step: {}, loss:{}, rmse: {}, mae: {}, step:{}, lr:{}, time:{}".format(use_fold, e, step,
                                                                                             running_loss / 1000,
                                                                                             l2.item(),
                                                                                             l3.item(),
                                                                                             c_step, c_lr,
                                                                                             end_time - start_time))
                running_loss = 0
                running_loss_c = 0
                start_time = time.time()
        print('******** waiting test **********')

        with torch.no_grad():
            net.eval()
            test_loss = 0
            l2p = 0
            l3p = 0
            for step, j in enumerate(test_dataloader):
                global_feature, trip_feature, sparse_feature, lengthes, label = j
                global_feature = global_feature.to(device)
                trip_feature = trip_feature.to(device)
                sparse_feature = sparse_feature.to(device)
                label = label.to(device)
                outs = net(sparse_feature, global_feature, trip_feature, lengthes)

                t_loss = criterion_regression(outs, label)
                l2 = rmse(outs, label)
                l3 = mae(outs, label)
                test_loss += t_loss.item()
                l2p += l2.item()
                l3p += l3.item()
            final_loss = test_loss / len(test_index)
            final_rmse = l2p / len(test_index)
            final_mae = l3p / len(test_index)
            # writer.add_scalar('use_fold_{}'.format(use_fold),final_loss,e)
            epoch_end = time.time()
            if final_loss < best_loss:
                best_loss = final_loss
            print('use_fold {}, test loss: {}, best loss: {}, rmse: {}, mae: {},  epoch time: {}'.format(use_fold, final_loss, best_loss,
                                                                                                         final_rmse, final_mae,
                                                                                      epoch_end - epoch_start))