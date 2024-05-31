"""
Baseline:
    T-GCN 
    GRU
    Chebnet
    Multi-head Self-attention
"""
import os
import time

import Metrics
import matplotlib.pyplot as plt
from T_GCN_model import GraphConv_T_GCN # model_T-GCN
from Chebnet_model import ChebNet # model_ChebNet
from GRU_model import GRUNet # model_GRU 
from Multi_head_attention import Attention


import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim

from torch.utils.data import DataLoader
from traffice_dataset_normal import LoadData # Data preprocessing file. Pems03 is different from other datasets and needs to be manually adjusted.

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    batch_size_ = 64
    
    data_set_num = 3 # The ID of dataset and 3 denotes PEMS03
    model_name = "Attention" # The name of model to be tested.
    
    file_csv = "data/PEMS0{}/PEMS0{}.csv".format(data_set_num, data_set_num) # The path of dataset's csv file
    file_npz = "data/PEMS0{}/PEMS0{}.npz".format(data_set_num, data_set_num) # The path of dataset's npz file
    file_result_save = "Baseline/{}/pems0{}/result.csv".format(model_name, data_set_num) # The storage path of result file
    file_model_save = "Baseline/{}/pems0{}/PEMS0{}_model".format(model_name, data_set_num, data_set_num) # The storage path of model file
    
    num_of_nodes = 358 #PEMS03 358 PEMS04 307 PEMS07 883 PEMS08 170
    train_data = LoadData(data_path=[file_csv, file_npz],
                          num_nodes=num_of_nodes,
                          divide_rate=[0.8, 0.2],
                          history_length=12,
                          pre_length = 12,
                          train_mode="train"
                          )

    train_loader = DataLoader(train_data, batch_size=batch_size_)
    test_data = LoadData(data_path=[file_csv, file_npz],
                          num_nodes=num_of_nodes,
                          divide_rate=[0.8, 0.2],
                          history_length=12,
                          pre_length = 12,
                          train_mode="test"
                          )
    
    test_loader = DataLoader(test_data, batch_size=batch_size_)
    
    device = torch.device("cuda:0")
    graph_data = train_data[0]["graph"].to(device)
    
    criterion = nn.MSELoss()
    result_temp = [[0,0,0,0,0,0]]
    name2 = ["num","mae", "rmse", "mape", "r2", "var"]
    test2 = pd.DataFrame(columns=name2, data=result_temp)
    test2.to_csv(file_result_save)
    # 模型训练
    Adam_Epoch = 400
    
    for i in [5]:
        print("------test{}-------".format(i))
        # my_net = GraphConv_T_GCN(in_c=12, hid_c=64, hidden_dim=100, final_output_dim=12) # T-GCN
        # my_net = ChebNet(in_c=12, hid_c=64, out_c=12, K=2) # ChebNet
        # my_net = GRUNet(input_dim=12, hidden_dim=64, final_output_dim=12) # GRU
        my_net = Attention(in_c=12,att_hiddem_size=64,final_output_dim=12) # Attention
        
        
        my_net = my_net.to(device)
        optimizer = optim.Adam(params=my_net.parameters()) # 默认学习率0.001
        
        loss_visio = []
        my_net.train() 
        for epoch in range(Adam_Epoch): 
            epoch_loss = 0.0 
            start_time = time.time()
            for data in train_loader: # data ["graph": [B, N, N], "flow_x":[B, N, H, D], "flow_y":[B, N, 1, D]]
                flow_x = data["flow_x"].to(device)    # [B, N, H, D]
                B, N= flow_x.size(0), flow_x.size(1)
                flow_x = flow_x.view(B, N, -1) # [B, N, H * D] H = 6 D = 1
                my_net.zero_grad() 
                predict_value = my_net(flow_x, graph_data).to(torch.device("cpu"))
                loss = criterion(predict_value, data["flow_y"]) 
                epoch_loss += loss.item()
                loss.backward() 
                optimizer.step()
            end_time = time.time()
            epoch_loss = 1000 * epoch_loss / len(train_data)
            loss_visio.append(epoch_loss)
            print("Adam Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch + 1, epoch_loss, (end_time - start_time) / 60))  
        torch.save(my_net, file_model_save + "_{}.pt".format(i))
    
        my_net.eval() 
        with torch.no_grad(): 
            num = 0
            all_predict_value = 0
            all_y_true = 0
            for data in test_loader:
                flow_x = data["flow_x"].to(device)    # [B, N, H, D]
                B, N= flow_x.size(0), flow_x.size(1)
                flow_x = flow_x.view(B, N, -1) # [B, N, H * D] H = 6 D = 1
                predict_value = my_net(flow_x, graph_data).to(torch.device("cpu"))
                if num == 0:
                    all_predict_value = predict_value
                    all_y_true = data["flow_y"]
                else:
                    all_predict_value = torch.cat([all_predict_value, predict_value], dim=0)
                    all_y_true =torch.cat([all_y_true, data["flow_y"]], dim=0)
                loss = criterion(predict_value, data["flow_y"])
                num += 1
        mae = Metrics.masked_mae_np(test_data.recover_data(test_data.flow_norm[0],test_data.flow_norm[1],all_y_true),test_data.recover_data(test_data.flow_norm[0],test_data.flow_norm[1],all_predict_value), 0)
        rmse = Metrics.masked_rmse_np(test_data.recover_data(test_data.flow_norm[0],test_data.flow_norm[1],all_y_true),test_data.recover_data(test_data.flow_norm[0],test_data.flow_norm[1],all_predict_value), 0)
        mape = Metrics.masked_mape_np(test_data.recover_data(test_data.flow_norm[0],test_data.flow_norm[1],all_y_true),test_data.recover_data(test_data.flow_norm[0],test_data.flow_norm[1],all_predict_value), 0)
        r2 = Metrics.r2_test(test_data.recover_data(test_data.flow_norm[0],test_data.flow_norm[1],all_y_true),test_data.recover_data(test_data.flow_norm[0],test_data.flow_norm[1],all_predict_value))
        var = Metrics.explained_variance_test(test_data.recover_data(test_data.flow_norm[0],test_data.flow_norm[1],all_y_true),test_data.recover_data(test_data.flow_norm[0],test_data.flow_norm[1],all_predict_value))
        print("my_net have {} paramerters in total".format(sum(x.numel() for x in my_net.parameters())))

        print("mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}, r2: {:02.4f}, var: {:02.4f}".format(mae, rmse, mape, r2, var)) 
        result_temp = [[i, mae, rmse, mape, r2.item(), var.item()]]
        test2 = pd.DataFrame(result_temp)
        test2.to_csv(file_result_save, mode="a", header=False)
        
        plt.title("train_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(loss_visio)
        plt.savefig(file_model_save + "_{}.png".format(i))
        plt.clf()
                       
                    
if __name__ == '__main__':
    main()
    
 

    
    