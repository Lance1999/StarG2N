'''
GRU Model
'''

import torch
import torch.nn as nn

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
        # inputs = inputs.view(batch_size, num_nodes, -1) # input [B, N, H_ * D]
        
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


class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, final_output_dim):
        super(GRUNet, self).__init__()
        
        self.model_name = "GRU"
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.gru_cell = GRUCell(self._input_dim, self._hidden_dim)
        self.linear_3 = nn.Linear(hidden_dim, final_output_dim)

    def forward(self, flow_x, graph_data):
        
        output_2 = flow_x.permute(0, 2, 1)
        batch_size, history_len, num_nodes = output_2.shape
        
        outputs = list()
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            output_2
        )
        
        for i in range(history_len):
            output, hidden_state = self.gru_cell(output_2[:, i, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
            outputs.append(output)
        last_output = outputs[-1]
        
        last_output = self.linear_3(last_output)
        last_output = last_output.unsqueeze(3)  #[batch node out_dim]
        
        return last_output
    
    def get_model_name(self):
        return self.model_name           
    