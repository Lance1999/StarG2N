import os
import time

import Metrics
import matplotlib.pyplot as plt
from model_v6 import GraphConv_v6

import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim

from torch.utils.data import DataLoader
from traffic_dataset_v5 import LoadData

def process_graph(graph_data):
    N = graph_data.size(0)
    matrix_i = torch.eye(N, dtype=graph_data.dtype, device=graph_data.device) 
    graph_data += matrix_i # A~  [N,N]
    degree_matrix = torch.sum(graph_data, dim=-1, keepdim=False) 
    degree_matrix = degree_matrix.pow(-1) 
    degree_matrix[degree_matrix == float("inf")] = 0.
    degree_matrix = torch.diag(degree_matrix) 
    return torch.mm(degree_matrix, graph_data) # D^(-1) * A = \hat(A) 

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    batch_size_ = 64
    train_data = LoadData(data_path=["/data/PEMS03/PEMS03.csv","/data/PEMS03/PEMS03.npz"],
                          num_nodes=358,
                          divide_rate=[0.8, 0.2],
                          history_length=12,
                          pre_length = 12,
                          train_mode="train"
                          )

    train_loader = DataLoader(train_data, batch_size=batch_size_)
    test_data = LoadData(data_path=["/data/PEMS03/PEMS03.csv","/data/PEMS03/PEMS03.npz"],
                          num_nodes=358,
                          divide_rate=[0.8, 0.2],
                          history_length=12,
                          pre_length = 12,
                          train_mode="test"
                          )
    
    test_loader = DataLoader(test_data, batch_size=batch_size_)
    device = torch.device("cuda:0")
    graph_data = train_data[0]["graph"].to(device)

    criterion = nn.MSELoss()
    result_temp = [[0,0,0,0,0]]
    name2 = ["mae", "rmse", "mape", "r2", "var"]
    test2 = pd.DataFrame(columns=name2, data=result_temp)
    test2.to_csv("/StarG2N/pems03/PEMS03_result.csv") # result file path
    
    Adam_Epoch = 400
    for i in range(5):
        print("------test{}-------".format(i))
        print(Adam_Epoch)
        my_net = GraphConv_v6(in_c=10, hid_c1=64, hid_c2=128, K=0, hidden_dim=128, att_hiddem_size=128, final_output_dim=12)
        my_net = my_net.to(device)
        optimizer = optim.Adam(params=my_net.parameters()) 
        
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
                predict_value = my_net(flow_x, graph_data, B, N).to(torch.device("cpu"))
                loss = criterion(predict_value, data["flow_y"]) 
                epoch_loss += loss.item()
                loss.backward() 
                optimizer.step() 
            end_time = time.time()
            epoch_loss = 1000 * epoch_loss / len(train_data)
            loss_visio.append(epoch_loss)
            print("Adam Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch + 1, epoch_loss, (end_time - start_time) / 60))  
        torch.save(my_net, "/STSCGNN/pems03/PEMS03_{}.pt".format(i))
       
        my_net.eval() 
        with torch.no_grad(): 
            num = 0
            all_predict_value = 0
            all_y_true = 0
            for data in test_loader:
                flow_x = data["flow_x"].to(device)    # [B, N, H, D]
                B, N= flow_x.size(0), flow_x.size(1)
                flow_x = flow_x.view(B, N, -1) # [B, N, H * D] H = 6 D = 1
                predict_value = my_net(flow_x, graph_data, B, N).to(torch.device("cpu"))
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
        
        print("mae: {:02.4f}, rmse: {:02.4f}, mape: {:02.4f}, r2: {:02.4f}, var: {:02.4f}".format(mae, rmse, mape, r2, var)) 
        result_temp = [[mae, rmse, mape, r2.item(), var.item()]]
        test2 = pd.DataFrame(result_temp)
        test2.to_csv("/StarG2N/pems03/PEMS03_result.csv",mode="a",header=False)
        
        plt.title("train_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(loss_visio)
        plt.savefig("/StarG2N/pems03/PEMS03_loss_{}.png".format(i))
        plt.clf()
                       
                    
if __name__ == '__main__':
    main()
 

    
    