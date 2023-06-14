import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Type_Embedding(torch.nn.Module):
    def __init__(self, seq_len, hidden_dim, output_dim):
        super(Type_Embedding, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.MLP1 = nn.Linear(self.seq_len, self.hidden_dim)
        self.ac = nn.GELU()
        self.MLP2 = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, x):
        x = self.MLP1(x)
        x = self.ac(x)
        x = self.MLP2(x)
        return x
    

class Separate_Embedding(torch.nn.Module):
    def __init__(self, seq_len, hidden_dim, output_dim):
        super(Separate_Embedding, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.linear_layers = nn.ModuleList([
            Type_Embedding(self.seq_len, self.hidden_dim,self.output_dim) for _ in range(10)
        ])
       
    def forward(self, x):
        tensor_list = torch.split(x, 1, dim=1)
        processed_tensors = []

        for i, tensor in enumerate(tensor_list):
            linear_layer = self.linear_layers[i]
            processed_tensor = linear_layer(tensor)
            processed_tensors.append(processed_tensor)
        
        combined_tensor = torch.cat(processed_tensors, dim=1)
        return combined_tensor
    
class Adj_Embedding(torch.nn.Module):
    def __init__(self, adj_matrix, hidden_dim_1, hidden_dim_2, output_dim, k):
        super(Adj_Embedding, self).__init__()
        self.adj_matrix = adj_matrix
        self.input_dim = self.adj_matrix.shape[0]
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.output_dim = output_dim
        self.k = k

        self.MLP1_d1 = nn.Linear(self.input_dim, self.hidden_dim_1)
        self.ac1_d1 = nn.GELU()
        self.MLP2_d1 = nn.Linear(self.hidden_dim_1, self.output_dim)

        self.MLP1_d2 = nn.Linear(self.input_dim, self.hidden_dim_2)
        self.ac1_d2 = nn.GELU()
        self.MLP2_d2 = nn.Linear(self.hidden_dim_2, self.output_dim)
    
    def forward(self, x):
        x = self.MLP1_d1(x)
        x = self.ac1_d1(x)
        x = x.reshape(x.shape[1], x.shape[0])
        x = self.MLP1_d2(x)
        x = self.ac1_d2(x)
        x = self.MLP2_d2(x)
        x = x.reshape(x.shape[1], x.shape[0])
        x = self.MLP2_d1(x)
        
        softmax_output = torch.nn.functional.softmax(x, dim=0)
        topk_values, topk_indices = torch.topk(softmax_output, self.k, dim=0)
        masked_output = torch.where(softmax_output < topk_values[-1], torch.tensor(0.0), softmax_output)

        return masked_output

class Linear(torch.nn.Module):
    def __init__(self, seq_len, hidden_dim, output_dim):
        super(Linear, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.MLP1 = nn.Linear(self.seq_len, self.hidden_dim)
        self.ac = nn.GELU()
        self.MLP2 = nn.Linear(self.hidden_dim, self.output_dim)
    
    def forward(self, x):
        x = self.MLP1(x)
        x = self.ac(x)
        x = self.MLP2(x)
        return x
    
class Framework(torch.nn.Module):
    def __init__(self, seq_len, hidden_dim_type, output_dim_type, adj_matrix, hidden_dim_1, hidden_dim_2, output_dim_adj, k, input_dim_main, hidden_dim_main,output_dim_main):
        super(Framework, self).__init__()
        self.seq_len = seq_len
        self.hidden_dim_type = hidden_dim_type
        self.output_dim_type = output_dim_type

        self.embedding = Separate_Embedding(self.seq_len, self.hidden_dim_type, self.output_dim_type)

        self.adj_matrix = adj_matrix
        self.matrix_dim = adj_matrix.shape[0]
        self.hidden_dim_1_adj = hidden_dim_1
        self.hidden_dim_2_adj = hidden_dim_2
        self.output_dim_adj = output_dim_adj
        self.k = k

        self.adj_embedding = Adj_Embedding(self.adj_matrix, self.hidden_dim_1_adj, self.hidden_dim_2_adj, self.output_dim_adj, self.k)

        self.input_dim_main = input_dim_main
        self.hidden_dim_main = hidden_dim_main
        self.output_dim_main = output_dim_main
        self.main = Linear(self.input_dim_main,self.hidden_dim_main,self.output_dim_main)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0,3,2,1)
        self.adj = self.adj_embedding(self.adj_matrix)
        
        x = self.main(x)
        x = x.permute(0,3,2,1)
        return x
