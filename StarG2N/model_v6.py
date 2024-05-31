import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import math
import torch.nn.functional as F

class ChebConv(nn.Module):
    
    def __init__(self, in_c, out_c, K, bias=True, normalize=True):
        super(ChebConv, self).__init__()
        self.normalize = normalize

        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))  # [K+1, 1, in_c, out_c]
        init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.K = K + 1

    def forward(self, inputs, graph):
        """
        :param inputs: the input data, [B, N, C]
        :param graph: the graph structure, [N, N]
        :return: convolution result, [B, N, D]
        """
        L = ChebConv.get_laplacian(graph, self.normalize)  # [N, N]
        mul_L = self.cheb_polynomial(L).unsqueeze(1)   # [K, 1, N, N]

        result = torch.matmul(mul_L, inputs)  # [K, B, N, C]
        result = torch.matmul(result, self.weight)  # [K, B, N, D]
        result = torch.sum(result, dim=0) + self.bias  # [B, N, D]

        return result
    
    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.

        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device, dtype=torch.float)  # [K, N, N]
        multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=torch.float)

        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.mm(laplacian, multi_order_laplacian[k-1]) - \
                                               multi_order_laplacian[k-2]

        return multi_order_laplacian
    
    def get_laplacian(graph, normalize):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L
    


class GRULinear(nn.Module):
    def __init__(self, num_gru_units: int, output_dim: int, bias: float=0.0):
        super(GRULinear, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)
        
    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape
        inputs = inputs.reshape((batch_size, num_nodes, 1)) # inputs (batch_size, num_nodes, 1)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        ) # hidden_state (batch_size, num_nodes, num_gru_units)
        
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        concatenation = concatenation.reshape((-1, self._num_gru_units + 1))
        outputs = concatenation @ self.weights + self.biases
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs
    
class GRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(GRUCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.linear1 = GRULinear(self._hidden_dim, self._hidden_dim * 2, bias=1.0)
        self.linear2 = GRULinear(self._hidden_dim, self._hidden_dim)
        
    def forward(self, inputs, hidden_state):
        concatenation = torch.sigmoid(self.linear1(inputs, hidden_state))
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        c = torch.tanh(self.linear2(inputs, r * hidden_state))
        new_hidden_state = u * hidden_state + (1 - u) * c
        return new_hidden_state, new_hidden_state

class selfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size):
        super(selfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0 :
            raise ValueError(
                "the hidden size %d is not a multiple of the number of attention heads"
                "%d" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.key_layer = nn.Linear(input_size, hidden_size)
        self.query_layer = nn.Linear(input_size, hidden_size)
        self.value_layer = nn.Linear(input_size, hidden_size)
        
    def trans_to_multiple_heads(self, x):
        new_size = x.size()[ : -1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_size)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, x):
        key = self.key_layer(x)
        query = self.query_layer(x)
        value = self.value_layer(x)

        key_heads = self.trans_to_multiple_heads(key)
        query_heads = self.trans_to_multiple_heads(query)
        value_heads = self.trans_to_multiple_heads(value)

        attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = F.softmax(attention_scores, dim = -1)

        context = torch.matmul(attention_probs, value_heads)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_size = context.size()[ : -2] + (self.all_head_size , )
        context = context.view(*new_size)
        return context

class GraphConv_v6(nn.Module):
    def __init__(self, in_c, hid_c1, hid_c2, K, hidden_dim, att_hiddem_size, final_output_dim):
        super(GraphConv_v6, self).__init__()
        
        self.model_name = "STSCGNN"
        self.conv1 = ChebConv(in_c=in_c, out_c=hid_c1, K=K)
        self.conv2 = ChebConv(in_c=hid_c1, out_c=hid_c2, K=K)
        self.act = nn.ReLU()

        self._input_dim = hid_c2 
        self._hidden_dim = hidden_dim
        self.gru_cell = GRUCell(self._input_dim, self._hidden_dim)
        
        self.num_attention_heads = 2
        self.attention_input = hidden_dim
        self.attention_hidden_size = att_hiddem_size
        self.attentionlayer = selfAttention(self.num_attention_heads, self.attention_input, self.attention_hidden_size)
        
        self.linear_3 = nn.Linear(self.attention_hidden_size, final_output_dim) 
 
    def forward(self, flow_x, graph_data, B, N):
        
       
        output_1 = self.act(self.conv1(flow_x, graph_data))
        output_1_2 = self.act(self.conv2(output_1, graph_data))

       
        num_nodes = N // 3
        output_2 = self.acc(output_1_2, num_nodes) 

        
        
        output_2 = output_2.permute(0, 2, 1)
        batch_size, history_len, num_nodes = output_2.shape
        
        outputs = list()
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            output_2
        )
        
        for i in range(history_len):
            output, hidden_state = self.gru_cell(output_2[:, i, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
            outputs.append(output)
        gpu_output = outputs[-1]   # [64, 307, 128]
        
       
        last_output = self.attentionlayer(gpu_output)
        
        
        last_output = self.linear_3(last_output)
        last_output = last_output.unsqueeze(3)
        return last_output
    
    def acc(self, output, num_nodes):
     
        output_temp01 = output[:,0:num_nodes,:] 
        output_temp02 = output[:,num_nodes:num_nodes*2,:]
        output_temp03 = output[:,num_nodes*2:num_nodes*3,:]
     
        output = torch.maximum(output_temp01, output_temp02)
        output =torch.maximum(output, output_temp03)
        return output

    def get_model_name(self):
        return self.model_name           
    